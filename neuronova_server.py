# ================================================================
# NEURONOVA — web backend for the Brain Tumour XAI project
#
# Serves the starfield web UI (neuronova_index.html) and exposes your
# existing tools as HTTP endpoints, one per pipeline stage:
#
#   POST /api/case                 Tool 1  - EfficientNetV2-B0 classification
#   POST /api/case/{id}/gradcam    Tool 2  - Grad-CAM
#   POST /api/case/{id}/hybrid     Tool 3  - EfficientNetV2-B0 + InceptionV3 soft vote
#   POST /api/case/{id}/stability  Tool 3B - test-time augmentation stability
#   POST /api/case/{id}/occlusion  Tool 3C - occlusion sensitivity vs Grad-CAM
#   POST /api/case/{id}/report     Tool 3D - findings / impression / reliability
#   POST /api/chat                 Tool 6  - Gemini assistant (falls back to Tool 4)
#
# Version 2 adds Tool 8 (no retraining needed):
#   POST /api/case/{id}/uncertainty  8A - Monte Carlo dropout + out-of-distribution check
#   POST /api/case/{id}/xai          8B - Grad-CAM++, Integrated Gradients, LIME, consensus
#   POST /api/case/{id}/similar      8C - most similar training scans
#   GET/POST /api/evaluation         8D - test-set metrics, ROC-AUC, calibration
#   POST /api/batch                  8E - many images at once
#   GET  /api/cases, /api/models     case history and a live model card
# and uses DenseNet121 as a third model when Tool 9 has trained it.
#
# The maths is copied from your notebook cells, so the numbers match
# what Tools 1-3D print. Nothing is retrained.
#
# Use from Colab:
#     import neuronova_server as nn
#     nn.launch(globals())
# ================================================================

import os
import io
import re
import sys
import json
import time
import uuid
import base64
import random
import shutil
import threading
import subprocess
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

VERSION = "3.0"

DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
EFFNET_FILES = ["BrainTumor_EfficientNetV2_B0.keras", "EfficientNetV2_B0.keras"]
INCEPTION_FILES = ["best_inceptionv3.keras", "BrainTumor_InceptionV3_Final.keras"]
DENSENET_FILES = ["BrainTumor_DenseNet121.keras"]
TEST_DIR_CANDIDATES = [
    "/content/brain_tumor_dataset/Testing",
    "brain_tumor_dataset/Testing",
]
TRAIN_DIR_CANDIDATES = [
    "/content/brain_tumor_dataset/Training",
    "brain_tumor_dataset/Training",
]
DRIVE_FOLDER = "/content/drive/MyDrive/brain_tumor_models"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

ALLOWED_EXT = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DISPLAY_MAX_SIDE = 640  # images sent to the browser are capped at this size

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


# ----------------------------------------------------------------
# Runtime state (filled by launch())
# ----------------------------------------------------------------

class _State:
    effnet = None
    inception = None
    class_names = list(DEFAULT_CLASS_NAMES)
    grad_model = None
    grad_model_logit = None
    grad_layer = None
    grad_layer_type = None
    gemini_client = None
    gemini_model = DEFAULT_GEMINI_MODEL
    gemini_types = None
    tool4_answer = None
    test_dir = None
    samples = []          # [{id, label, path}]
    cases = {}            # case_id -> dict
    lock = threading.Lock()
    html_path = None
    densenet = None
    sources = {}          # model key -> where it was loaded from
    model_paths = {}      # model key -> .keras file (for the size column of the benchmark)
    train_dir = None
    splits = {}           # model key -> (feature model, head layers)
    index = None          # features of training scans (Tool 8A/8C)
    index_status = {"state": "idle"}
    eval_result = None    # Tool 8D results
    eval_status = {"state": "idle"}
    unet = None           # Tool 10 U-Net
    unet_meta = None
    unet_path = None
    xai_mask_eval = None  # Tool 10B
    volumes = {}          # Tool 11: volume id -> arrays
    rag = None            # Tool 13 knowledge base
    rag_status = {"state": "idle"}
    store = None          # case management database
    auth_enabled = False
    research = {}         # Tool 12 results
    lab_status = {}
    ready = False


S = _State()


# ----------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------

def _png_data_url(img, fmt="PNG"):
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _display_size(w, h, cap=DISPLAY_MAX_SIDE):
    scale = min(1.0, cap / float(max(w, h)))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _jet(values):
    """matplotlib 'jet' colormap without importing matplotlib."""
    v = np.clip(np.asarray(values, dtype=np.float32), 0, 1)

    def interp(points):
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        return np.interp(v, xs, ys)

    r = interp([(0, 0), (0.35, 0), (0.66, 1), (0.89, 1), (1, 0.5)])
    g = interp([(0, 0), (0.125, 0), (0.375, 1), (0.64, 1), (0.91, 0), (1, 0)])
    b = interp([(0, 0.5), (0.11, 1), (0.34, 1), (0.65, 0), (1, 0)])
    return np.stack([r, g, b], axis=-1)


def _overlay(base_rgb_uint8, values, alpha=0.45):
    base = base_rgb_uint8.astype(np.float32) / 255.0
    col = _jet(values)
    out = np.clip((1 - alpha) * base + alpha * col, 0, 1)
    return Image.fromarray(np.uint8(out * 255))


def _ms(t0):
    return int(round((time.perf_counter() - t0) * 1000))


def _case(case_id):
    c = S.cases.get(case_id)
    if c is None:
        return load_case(case_id)      # saved by an earlier session
    return c


def _remember(case):
    S.cases[case["case_id"]] = case
    # keep the 20 most recent cases in memory
    if len(S.cases) > 20:
        for old in sorted(S.cases.values(), key=lambda c: c["t"])[:-20]:
            S.cases.pop(old["case_id"], None)


# ----------------------------------------------------------------
# Models (reuse notebook variables when present, else load files)
# ----------------------------------------------------------------

def _input_hw(model):
    try:
        shp = model.input_shape
        if isinstance(shp, list):
            shp = shp[0]
        return int(shp[1]), int(shp[2])
    except Exception:
        return None


def _load_first(paths, search_dirs):
    from tensorflow.keras.models import load_model
    for d in search_dirs:
        for p in paths:
            full = os.path.join(d, p)
            if os.path.exists(full):
                return load_model(full), full
    return None, None


def _setup_models(ns):
    search_dirs = [os.getcwd(), "/content", _HERE]

    # EfficientNetV2-B0: Tool 1 calls it `model`; Tool 3 calls it `effnet_model`
    eff = None
    for name in ("effnet_model", "model"):
        m = ns.get(name)
        if m is not None and hasattr(m, "predict") and _input_hw(m) == (224, 224):
            eff = m
            S.sources["efficientnet"] = f"notebook variable `{name}`"
            print(f"  EfficientNetV2-B0 : notebook variable `{name}`")
            break
    if eff is None:
        eff, path = _load_first(EFFNET_FILES, search_dirs)
        if eff is None:
            raise RuntimeError(
                "EfficientNetV2-B0 not found. Run the cell that loads "
                "BrainTumor_EfficientNetV2_B0.keras first, or put that file in /content."
            )
        S.sources["efficientnet"] = os.path.basename(path)
        print(f"  EfficientNetV2-B0 : {path}")

    inc = ns.get("inception_model")
    if inc is not None and _input_hw(inc) == (299, 299):
        S.sources["inception"] = "notebook variable `inception_model`"
        print("  InceptionV3       : notebook variable `inception_model`")
    else:
        inc, path = _load_first(INCEPTION_FILES, search_dirs + [DRIVE_FOLDER])
        if inc is not None:
            S.sources["inception"] = os.path.basename(path)
            print(f"  InceptionV3       : {path}")
        else:
            print("  InceptionV3       : not found, hybrid runs in single-model mode")

    names = ns.get("class_names")
    if isinstance(names, (list, tuple)) and len(names) == 4:
        S.class_names = [str(x) for x in names]
    else:
        S.class_names = list(DEFAULT_CLASS_NAMES)
    print("  classes           :", S.class_names)

    dn = ns.get("densenet_model")
    if dn is not None and _input_hw(dn) == (224, 224):
        S.sources["densenet"] = "notebook variable `densenet_model`"
        print("  DenseNet121       : notebook variable `densenet_model`")
    else:
        dn, path = _load_first(DENSENET_FILES, search_dirs + [DRIVE_FOLDER])
        if dn is not None:
            S.sources["densenet"] = os.path.basename(path)
            print(f"  DenseNet121       : {path} (third model from Tool 9)")

    S.effnet = eff
    S.inception = inc
    S.densenet = dn
    S.model_paths = {}
    for key, names_ in (("efficientnet", EFFNET_FILES), ("inception", INCEPTION_FILES), ("densenet", DENSENET_FILES)):
        for d in search_dirs + [DRIVE_FOLDER]:
            hit = next((os.path.join(d, n) for n in names_ if os.path.exists(os.path.join(d, n))), None)
            if hit:
                S.model_paths[key] = hit
                break
    S.grad_model = None
    S.grad_model_logit = None
    S.splits = {}


def _setup_gemini(ns):
    S.gemini_client = None
    S.gemini_model = ns.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    try:
        from google.genai import types as gtypes
        S.gemini_types = gtypes
    except Exception:
        print("  Gemini            : google-genai not installed (run: %pip install -q -U google-genai)")
        return

    cl = ns.get("client")
    if cl is not None and hasattr(cl, "models"):
        S.gemini_client = cl
        print(f"  Gemini            : Tool 6 client, model {S.gemini_model}")
        return

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if key and key.strip():
        try:
            from google import genai
            S.gemini_client = genai.Client(api_key=key.strip())
            print(f"  Gemini            : key from secrets, model {S.gemini_model}")
        except Exception as exc:
            print("  Gemini            : could not create client:", exc)
    else:
        print("  Gemini            : no GEMINI_API_KEY, assistant uses offline answers")


def _setup_samples():
    S.samples = []
    S.test_dir = None
    for d in TEST_DIR_CANDIDATES:
        if os.path.isdir(d):
            S.test_dir = d
            break
    if not S.test_dir:
        print("  Samples           : Testing folder not found (sample picker hidden)")
        return
    rng = random.Random(7)
    for cls in sorted(os.listdir(S.test_dir)):
        folder = os.path.join(S.test_dir, cls)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXT))
        for f in rng.sample(files, min(2, len(files))):
            S.samples.append({"id": f"{cls}/{f}", "label": cls, "path": os.path.join(folder, f)})
    print(f"  Samples           : {len(S.samples)} images from {S.test_dir}")


# ----------------------------------------------------------------
# Prediction helpers (identical preprocessing to Tool 1 / Tool 3)
#   EfficientNetV2-B0 : 224x224, raw 0-255
#   InceptionV3       : 299x299, rescaled to 0-1
# ----------------------------------------------------------------

def _prep_eff(pil):
    return np.array(pil.convert("RGB").resize((224, 224)), dtype=np.float32)


def _prep_inc(pil):
    return np.array(pil.convert("RGB").resize((299, 299)), dtype=np.float32) / 255.0


def _run(model, batch, chunk=32):
    outs = []
    for s in range(0, len(batch), chunk):
        outs.append(np.asarray(model(batch[s:s + chunk], training=False), dtype=np.float32))
    return np.concatenate(outs, axis=0)


def predict_effnet(pils):
    return _run(S.effnet, np.stack([_prep_eff(p) for p in pils]))


def predict_inception(pils):
    return _run(S.inception, np.stack([_prep_inc(p) for p in pils]))


def predict_densenet(pils):
    return _run(S.densenet, np.stack([_prep_eff(p) for p in pils]))


def predict_hybrid(pils, w_eff=0.5):
    p_eff = predict_effnet(pils)
    if S.inception is None:
        return p_eff, p_eff, None
    p_inc = predict_inception(pils)
    return w_eff * p_eff + (1.0 - w_eff) * p_inc, p_eff, p_inc


def entropy_normalized(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def js_divergence(p, q):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a * np.log(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _probs_dict(p):
    return {c: float(p[i] * 100) for i, c in enumerate(S.class_names)}


# ----------------------------------------------------------------
# Stage 1 — Tool 1 classification
# ----------------------------------------------------------------

def stage_classify(case):
    t0 = time.perf_counter()
    img = case["image"]
    arr = np.array(img)
    mean_b = float(arr.mean())
    std_b = float(arr.std())
    warnings = []
    if mean_b < 15:
        warnings.append("Image is extremely dark.")
    if mean_b > 245:
        warnings.append("Image is extremely bright.")
    if std_b < 10:
        warnings.append("Image has very low visual variation.")

    probs = predict_effnet([img])[0]
    if len(probs) != len(S.class_names):
        raise ValueError(f"Model outputs {len(probs)} classes but there are {len(S.class_names)} class names.")
    idx = int(np.argmax(probs))
    conf = float(probs[idx] * 100)

    if conf >= 90:
        level = "High model confidence"
    elif conf >= 70:
        level = "Moderate model confidence"
    elif conf >= 50:
        level = "Low model confidence"
    else:
        level = "Very low model confidence"

    if conf < 70:
        warn = ("The model has relatively low confidence. "
                "This AI prediction should not be relied on for clinical decisions.")
    else:
        warn = ("The model has relatively high confidence. "
                "This is an AI prediction, not a confirmed medical diagnosis.")

    top = [{"rank": r + 1, "class": S.class_names[i], "probability": float(probs[i] * 100)}
           for r, i in enumerate(np.argsort(probs)[::-1][:3])]

    case["eff_probs"] = probs
    return {
        "model": "EfficientNetV2-B0",
        "input": "224 x 224 x 3",
        "prediction": S.class_names[idx],
        "index": idx,
        "confidence": conf,
        "confidence_level": level,
        "probabilities": _probs_dict(probs),
        "raw_output": [float(x) for x in probs],
        "top3": top,
        "quality": {"mean_brightness": mean_b, "brightness_std": std_b, "warnings": warnings},
        "uncertainty_warning": warn,
        "ms": _ms(t0),
    }


# ----------------------------------------------------------------
# Stage 2 — Tool 2 Grad-CAM
# ----------------------------------------------------------------

def _ensure_grad_model():
    if S.grad_model is not None:
        return
    import tensorflow as tf
    candidates = []
    for layer in S.effnet.layers:
        try:
            shp = layer.output.shape
            if len(shp) == 4 and shp[1] is not None and shp[2] is not None:
                candidates.append(layer)
        except Exception:
            continue
    if not candidates:
        raise ValueError("Could not find a 4-D convolutional layer for Grad-CAM.")
    target = candidates[-1]  # deepest, same rule as Tool 2
    S.grad_layer = target.name
    S.grad_layer_type = type(target).__name__
    S.grad_model = tf.keras.models.Model(inputs=S.effnet.inputs, outputs=[target.output, S.effnet.output])


def _ensure_logit_model():
    """Same layer as Tool 2, but returns the input of the last Dense layer so the pre-softmax class score
    (the score used in the original Grad-CAM paper) can be formed. Used only as a fallback."""
    if getattr(S, "grad_model_logit", None) is not None:
        return S.grad_model_logit
    import tensorflow as tf
    _ensure_grad_model()
    last = S.effnet.layers[-1]
    act = getattr(getattr(last, "activation", None), "__name__", "")
    if type(last).__name__ != "Dense" or act != "softmax":
        S.grad_model_logit = False
        return False
    target = S.effnet.get_layer(S.grad_layer)
    m = tf.keras.models.Model(inputs=S.effnet.inputs, outputs=[target.output, last.input])
    S.grad_model_logit = (m, last.kernel, last.bias)
    return S.grad_model_logit


def _gradcam_raw(x, idx, use_logit=False):
    import tensorflow as tf
    if use_logit:
        lm = _ensure_logit_model()
        if not lm:
            return None, 0.0
        m, W_, b_ = lm
        with tf.GradientTape() as tape:
            conv_out, h = m(x, training=False)
            score = tf.matmul(h, W_)[:, idx] + b_[idx]
    else:
        with tf.GradientTape() as tape:
            conv_out, preds = S.grad_model(x, training=False)
            score = preds[:, idx]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Gradients could not be calculated for Grad-CAM.")
    pooled = tf.reduce_mean(grads, axis=(1, 2))[0]
    heat = tf.reduce_sum(conv_out[0] * pooled[tf.newaxis, tf.newaxis, :], axis=-1)
    heat = tf.maximum(heat, 0).numpy()
    return heat, float(heat.max())


def _finite(o):
    """Replace NaN / infinity with None so a result can always be sent as JSON."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def stage_gradcam(case):
    import tensorflow as tf
    t0 = time.perf_counter()
    if "classify" not in case["results"]:
        case["results"]["classify"] = stage_classify(case)
    idx = case["results"]["classify"]["index"]
    img = case["image"]
    _ensure_grad_model()

    x = tf.cast(np.expand_dims(_prep_eff(img), 0), tf.float32)
    heat, mx = _gradcam_raw(x, idx)
    score_used = "softmax probability (as in Tool 2)"
    if mx <= 0:
        # A saturated or out-of-distribution prediction can leave no positive evidence in the softmax
        # gradients. Retry with the pre-softmax score, which is what the original Grad-CAM paper uses.
        heat2, mx2 = _gradcam_raw(x, idx, use_logit=True)
        if mx2 > 0:
            heat, mx = heat2, mx2
            score_used = "pre-softmax score (fallback; the softmax map was empty)"
    empty = mx <= 0
    heat = np.zeros_like(heat, dtype=np.float32) if empty else (heat / mx).astype(np.float32)

    W, H = img.size
    heat_img = Image.fromarray(np.uint8(heat * 255)).resize((W, H), Image.Resampling.BILINEAR)
    heat_resized = np.array(heat_img, dtype=np.float32) / 255.0
    case["heatmap_resized"] = heat_resized

    thr = 0.60
    mask = heat_resized >= thr
    act_pct = float(mask.sum() / mask.size * 100)
    ys, xs = np.where(mask)
    bbox = None
    if len(xs):
        bbox = {"x_min": int(xs.min()), "y_min": int(ys.min()), "x_max": int(xs.max()),
                "y_max": int(ys.max()), "width": int(xs.max() - xs.min()), "height": int(ys.max() - ys.min())}

    mean_a = float(heat_resized.mean())
    max_a = float(heat_resized.max())
    if empty:
        intensity = "No positive evidence"
    elif mean_a >= 0.50:
        intensity = "Strong overall activation"
    elif mean_a >= 0.25:
        intensity = "Moderate overall activation"
    else:
        intensity = "Low overall activation"

    cls = case["results"]["classify"]
    explanation = (
        f"The EfficientNetV2-B0 model predicted '{cls['prediction']}' with a confidence of "
        f"{cls['confidence']:.2f}%. The Grad-CAM visualization highlights image regions that "
        f"contributed strongly to this prediction. Approximately {act_pct:.2f}% of the image was "
        f"above the selected activation threshold. The highlighted region represents model attention "
        f"and should not be interpreted as a definitive tumor boundary or medical diagnosis."
    )

    if empty:
        explanation = (
            f"The EfficientNetV2-B0 model predicted '{cls['prediction']}' with a confidence of "
            f"{cls['confidence']:.2f}%, but Grad-CAM found no image region with positive evidence for that class "
            f"in the {S.grad_layer} layer, so the map is empty. This often happens when an image is very different "
            f"from the training scans. The prediction has no visual explanation and should be treated with extra caution."
        )

    dw, dh = case["display_size"]
    gray = heat_img.resize((dw, dh), Image.Resampling.BILINEAR) if (dw, dh) != (W, H) else heat_img
    return {
        "layer": S.grad_layer,
        "layer_type": S.grad_layer_type,
        "heatmap_gray": _png_data_url(gray.convert("L")),
        "activation_threshold": thr,
        "activation_percent": act_pct,
        "mean_activation": mean_a,
        "max_activation": max_a,
        "intensity": intensity,
        "bbox": bbox,
        "empty": bool(empty),
        "score_used": score_used,
        "original_size": [W, H],
        "explanation": explanation,
        "ms": _ms(t0),
    }


# ----------------------------------------------------------------
# Stage 3 — Tool 3 hybrid ensemble
# ----------------------------------------------------------------

def stage_hybrid(case):
    t0 = time.perf_counter()
    hyb_all, eff_all, inc_all = predict_hybrid([case["image"]])
    eff, hyb = eff_all[0], hyb_all[0]
    inc = inc_all[0] if inc_all is not None else None
    enabled = inc is not None

    eff_i, hyb_i = int(np.argmax(eff)), int(np.argmax(hyb))
    inc_i = int(np.argmax(inc)) if enabled else None
    hyb_conf = float(hyb[hyb_i] * 100)
    ent = entropy_normalized(hyb)

    if enabled:
        agree = eff_i == inc_i
        jsd = js_divergence(eff, inc)
        score = float(max(0.0, 1.0 - jsd / np.log(2)) * 100)
    else:
        agree, jsd, score = True, 0.0, 100.0

    if not enabled:
        status = "SINGLE MODEL (InceptionV3 not available)"
    elif agree and hyb_conf >= 85:
        status = "STRONG AGREEMENT"
    elif agree:
        status = "AGREEMENT, MODERATE CONFIDENCE"
    else:
        status = "DISAGREEMENT — REVIEW NEEDED"

    case["hybrid_probs"] = hyb
    extra = {}
    if S.densenet is not None:
        dn = predict_densenet([case["image"]])[0]
        dn_i = int(np.argmax(dn))
        extra["densenet"] = {"prediction": S.class_names[dn_i], "confidence": float(dn[dn_i] * 100),
                             "probabilities": _probs_dict(dn)}
        if enabled:
            e3 = (eff + inc + dn) / 3.0
            e3_i = int(np.argmax(e3))
            extra["ensemble3"] = {"prediction": S.class_names[e3_i], "confidence": float(e3[e3_i] * 100),
                                  "probabilities": _probs_dict(e3), "entropy": entropy_normalized(e3),
                                  "all_agree": bool(eff_i == inc_i == dn_i)}
    rs = reliability_settings()
    dm = "hybrid" if enabled else "efficientnet"
    T = float(rs["temperatures"].get(dm, 1.0))
    pc = temper(hyb, T)
    pc_i = int(np.argmax(pc))
    extra["calibrated"] = {"temperature": T, "source": rs["source"], "prediction": S.class_names[pc_i],
                           "confidence": float(pc[pc_i] * 100), "probabilities": _probs_dict(pc),
                           "threshold": rs["threshold"] * 100}
    return {
        **extra,
        "hybrid_enabled": enabled,
        "efficientnet": {"prediction": S.class_names[eff_i], "confidence": float(eff[eff_i] * 100),
                         "probabilities": _probs_dict(eff)},
        "inception": ({"prediction": S.class_names[inc_i], "confidence": float(inc[inc_i] * 100),
                       "probabilities": _probs_dict(inc)} if enabled else None),
        "hybrid": {"prediction": S.class_names[hyb_i], "confidence": hyb_conf,
                   "probabilities": _probs_dict(hyb), "entropy": ent},
        "agreement": {"models_agree": bool(agree), "agreement_score": score,
                      "js_divergence": float(jsd), "status": status},
        "ms": _ms(t0),
    }


# ----------------------------------------------------------------
# Stage 4 — Tool 3B stability (test-time augmentation)
# ----------------------------------------------------------------

def _zoom_center(img, factor):
    w, h = img.size
    cw, ch = int(w / factor), int(h / factor)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h))


AUGMENTATIONS = [
    ("original", lambda im: im),
    ("horizontal flip", lambda im: ImageOps.mirror(im)),
    ("rotate +8°", lambda im: im.rotate(8, resample=Image.Resampling.BILINEAR)),
    ("rotate -8°", lambda im: im.rotate(-8, resample=Image.Resampling.BILINEAR)),
    ("zoom 1.10x", lambda im: _zoom_center(im, 1.10)),
    ("zoom 1.20x", lambda im: _zoom_center(im, 1.20)),
    ("brightness +12%", lambda im: ImageEnhance.Brightness(im).enhance(1.12)),
    ("brightness -12%", lambda im: ImageEnhance.Brightness(im).enhance(0.88)),
    ("contrast +12%", lambda im: ImageEnhance.Contrast(im).enhance(1.12)),
    ("contrast -12%", lambda im: ImageEnhance.Contrast(im).enhance(0.88)),
]


def stage_stability(case):
    t0 = time.perf_counter()
    if "hybrid_probs" not in case:
        case["results"]["hybrid"] = stage_hybrid(case)
    ref = int(np.argmax(case["hybrid_probs"]))
    names = [n for n, _ in AUGMENTATIONS]
    imgs = [f(case["image"]) for _, f in AUGMENTATIONS]
    aug_h, _, _ = predict_hybrid(imgs)
    pred_idx = np.argmax(aug_h, axis=1)

    consistency = float(np.mean(pred_idx == ref) * 100)
    ref_conf = aug_h[:, ref] * 100
    c_mean, c_std, c_min = float(ref_conf.mean()), float(ref_conf.std()), float(ref_conf.min())

    if consistency == 100 and c_std < 5:
        level = "VERY STABLE"
    elif consistency >= 80 and c_std < 10:
        level = "STABLE"
    elif consistency >= 60:
        level = "MODERATELY STABLE"
    else:
        level = "UNSTABLE"

    rows = []
    for n, i, row, rc in zip(names, pred_idx, aug_h, ref_conf):
        rows.append({"name": n, "prediction": S.class_names[int(i)], "top_confidence": float(row.max() * 100),
                     "ref_confidence": float(rc), "changed": bool(int(i) != ref)})

    return {
        "reference": S.class_names[ref],
        "augmentations": rows,
        "augmentations_tested": len(names),
        "consistency_percent": consistency,
        "confidence_mean": c_mean,
        "confidence_std": c_std,
        "confidence_min": c_min,
        "level": level,
        "changed_under": [r["name"] for r in rows if r["changed"]],
        "ms": _ms(t0),
    }


# ----------------------------------------------------------------
# Stage 5 — Tool 3C occlusion sensitivity vs Grad-CAM
# ----------------------------------------------------------------

def stage_occlusion(case, size=224, patch=40, stride=20, batch=32, top_fraction=0.20):
    t0 = time.perf_counter()
    if "hybrid_probs" not in case:
        case["results"]["hybrid"] = stage_hybrid(case)
    if "heatmap_resized" not in case:
        try:
            case["results"]["gradcam"] = stage_gradcam(case)
        except Exception:
            pass

    hyb = case["hybrid_probs"]
    ref = int(np.argmax(hyb))
    ref_p = float(hyb[ref])
    base_img = case["image"].resize((size, size))
    base = np.array(base_img, dtype=np.uint8)
    fill = int(base.mean())

    positions = [(r, c) for r in range(0, size - patch + 1, stride) for c in range(0, size - patch + 1, stride)]
    drop_sum = np.zeros((size, size), dtype=np.float64)
    count = np.zeros((size, size), dtype=np.float64)
    for s in range(0, len(positions), batch):
        chunk = positions[s:s + batch]
        imgs = []
        for (r, c) in chunk:
            a = base.copy()
            a[r:r + patch, c:c + patch, :] = fill
            imgs.append(Image.fromarray(a))
        probs, _, _ = predict_hybrid(imgs)
        drops = ref_p - probs[:, ref]
        for (r, c), d in zip(chunk, drops):
            drop_sum[r:r + patch, c:c + patch] += d
            count[r:r + patch, c:c + patch] += 1

    occ = np.clip(drop_sum / np.maximum(count, 1), 0, None)
    occ_norm = occ / occ.max() if occ.max() > 0 else occ.copy()
    max_drop = float(occ.max() * 100)

    thr = np.quantile(occ_norm, 1 - top_fraction)
    occ_mask = occ_norm >= max(thr, 1e-6)
    ys, xs = np.where(occ_mask)
    occ_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None

    iou = corr = None
    agreement = "NOT AVAILABLE (Grad-CAM heatmap not found)"
    g_overlay = overlap = None
    if "heatmap_resized" in case:
        g = np.array(Image.fromarray(np.uint8(np.clip(case["heatmap_resized"], 0, 1) * 255)).resize(
            (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        g_norm = g / g.max() if g.max() > 0 else g
        g_thr = np.quantile(g_norm, 1 - top_fraction)
        g_mask = g_norm >= max(g_thr, 1e-6)
        inter = np.logical_and(g_mask, occ_mask).sum()
        union = np.logical_or(g_mask, occ_mask).sum()
        iou = float(inter / union) if union > 0 else 0.0
        corr = (float(np.corrcoef(occ_norm.ravel(), g_norm.ravel())[0, 1])
                if occ_norm.std() > 0 and g_norm.std() > 0 else 0.0)
        if iou >= 0.50 and corr >= 0.50:
            agreement = "HIGH AGREEMENT"
        elif iou >= 0.25 or corr >= 0.30:
            agreement = "MODERATE AGREEMENT"
        else:
            agreement = "LOW AGREEMENT"
        g_overlay = _png_data_url(_overlay(base, g_norm))
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[..., 0] = g_mask.astype(np.uint8) * 255     # red    = Grad-CAM only
        rgb[..., 1] = occ_mask.astype(np.uint8) * 255   # green  = occlusion only, yellow = both
        overlap = _png_data_url(Image.fromarray(rgb))

    return {
        "patch": patch,
        "stride": stride,
        "positions": len(positions),
        "reference": {"class": S.class_names[ref], "probability": ref_p * 100},
        "max_confidence_drop_points": max_drop,
        "important_region_bbox": occ_bbox,
        "gradcam_iou": iou,
        "gradcam_correlation": corr,
        "agreement_level": agreement,
        "occlusion_overlay": _png_data_url(_overlay(base, occ_norm)),
        "gradcam_overlay": g_overlay,
        "overlap_mask": overlap,
        "ms": _ms(t0),
    }


# ----------------------------------------------------------------
# Stage 6 — Tool 3D report + reliability
# ----------------------------------------------------------------

def build_report(hybrid, stability=None, occlusion=None):
    """Port of Tool 3D. Takes the stage results (dicts) and returns the report."""
    hyb = hybrid["hybrid"]
    agr = hybrid["agreement"]
    stab = stability
    occ = occlusion
    enabled = hybrid["hybrid_enabled"]

    findings, impression, diagnosis, recommendations, measurements = [], [], [], [], []

    if enabled:
        findings.append(f"Hybrid CNN (EfficientNetV2-B0 + InceptionV3) predicts: {hyb['prediction']} "
                        f"with {hyb['confidence']:.2f}% confidence.")
        findings.append(f"EfficientNetV2-B0 predicts {hybrid['efficientnet']['prediction']} "
                        f"({hybrid['efficientnet']['confidence']:.2f}%); InceptionV3 predicts "
                        f"{hybrid['inception']['prediction']} ({hybrid['inception']['confidence']:.2f}%).")
        findings.append(f"Model agreement: {agr['status']} (agreement score {agr['agreement_score']:.1f}/100).")
    else:
        findings.append(f"EfficientNetV2-B0 predicts: {hyb['prediction']} with {hyb['confidence']:.2f}% "
                        f"confidence (single-model mode).")
    findings.append(f"Normalised prediction entropy: {hyb['entropy']:.3f} (0 = certain, 1 = uncertain).")

    if stab:
        findings.append(f"Stability test over {stab['augmentations_tested']} image variations: {stab['level']} "
                        f"({stab['consistency_percent']:.0f}% same prediction, confidence "
                        f"{stab['confidence_mean']:.1f}% +/- {stab['confidence_std']:.1f}).")
        if stab["changed_under"]:
            findings.append("Prediction changed under: " + ", ".join(stab["changed_under"]) + ".")

    if occ:
        if occ["gradcam_iou"] is not None:
            findings.append(f"Occlusion sensitivity vs Grad-CAM: {occ['agreement_level']} (overlap IoU "
                            f"{occ['gradcam_iou']:.2f}, correlation {occ['gradcam_correlation']:.2f}).")
        else:
            findings.append("Occlusion sensitivity computed; Grad-CAM comparison not available.")
        findings.append(f"Hiding the most influential image patch reduced model confidence by "
                        f"{occ['max_confidence_drop_points']:.1f} percentage points.")
        if occ["important_region_bbox"]:
            measurements.append("Model-attention region (occlusion, 224x224 pixel box [x1,y1,x2,y2]): "
                                f"{occ['important_region_bbox']}")

    checks = [
        {"name": "Both CNNs predict the same class", "passed": bool(agr["models_agree"]), "counted": True},
        {"name": "Hybrid confidence is at least 85%", "passed": hyb["confidence"] >= 85, "counted": True},
        {"name": "Same class in at least 80% of image variations",
         "passed": bool(stab and stab["consistency_percent"] >= 80), "counted": bool(stab)},
        {"name": "Occlusion and Grad-CAM show high agreement",
         "passed": bool(occ and occ["agreement_level"] == "HIGH AGREEMENT"),
         "counted": bool(occ and occ["gradcam_iou"] is not None)},
    ]
    points = sum(1 for c in checks if c["passed"] and c["counted"])
    total = sum(1 for c in checks if c["counted"])
    ratio = points / total if total else 0
    label = "HIGH" if ratio >= 0.80 else ("MODERATE" if ratio >= 0.50 else "LOW")

    impression.append(f"AI result reliability: {label} ({points}/{total} checks passed).")
    impression.append(f"The hybrid model favours the class '{hyb['prediction']}'.")
    diagnosis.append(f"AI-predicted class: {hyb['prediction']} (model output only, not a clinical diagnosis).")

    recommendations.append("Have a radiologist/neurologist review the MRI; AI output must not replace clinical judgement.")
    if label != "HIGH":
        recommendations.append("Reliability is not high: treat this AI result with extra caution and seek expert review.")
    if not agr["models_agree"]:
        recommendations.append("The two CNNs disagree: manual expert review is strongly advised.")
    if stab and stab["level"] in ("UNSTABLE", "MODERATELY STABLE"):
        recommendations.append("Prediction is sensitive to small image changes: use a higher-quality scan if possible.")

    clinical = ["No patient clinical history was supplied to the AI. Prediction is based only on the MRI image."]

    lines = ["AI-GENERATED HYBRID CNN ANALYSIS SUMMARY",
             "(Generated by the AI models. This is NOT a radiologist's report.)", "", "FINDINGS:"]
    lines += [f"- {x}" for x in findings]
    lines += ["", "IMPRESSION:"] + [f"- {x}" for x in impression]
    lines += ["", "DIAGNOSIS:"] + [f"- {x}" for x in diagnosis]
    lines += ["", "RECOMMENDATIONS:"] + [f"- {x}" for x in recommendations]
    if measurements:
        lines += ["", "MEASUREMENTS:"] + [f"- {x}" for x in measurements]

    return {
        "findings": findings,
        "impression": impression,
        "diagnosis": diagnosis,
        "recommendations": recommendations,
        "measurements": measurements,
        "clinical_information": clinical,
        "reliability": {"label": label, "checks_passed": points, "checks_total": total, "checks": checks},
        "summary_text": "\n".join(lines),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def stage_report(case):
    t0 = time.perf_counter()
    res = case["results"]
    if "hybrid" not in res:
        res["hybrid"] = stage_hybrid(case)
    rep = build_report(res["hybrid"], res.get("stability"), res.get("occlusion"))
    sg = res.get("segment") or {}
    if sg.get("available"):
        rep["segmentation"] = {k: sg.get(k) for k in ("detected", "area_px", "area_percent_image", "area_percent_brain",
                                                      "area_mm2", "bbox", "location", "mean_probability",
                                                      "consistent_with_classifier")}
        lines = ["", f"SEGMENTATION (TOOL 10, {sg['model']['arch']}):", "- " + sg["summary"]]
        if sg["detected"]:
            m_line = (f"U-Net tumour region: {sg['area_px']:,} px ({sg['area_percent_image']:.2f}% of the image"
                      + (f", {sg['area_percent_brain']:.2f}% of the brain area" if sg.get("area_percent_brain") else "")
                      + (f", {sg['area_mm2']:.0f} mm2" if sg.get("area_mm2") else "")
                      + f"), box [x1,y1,x2,y2] = {sg['bbox']}")
            rep["measurements"] = list(rep.get("measurements") or []) + [m_line]
            lines.append("- " + m_line + ".")
            g = sg.get("gradcam_vs_mask")
            if g:
                lines.append(f"- Grad-CAM peak {'inside' if g['peak_inside'] else 'outside'} the segmented region; "
                             f"{g['energy_inside'] * 100:.0f}% of Grad-CAM energy falls inside it.")
        if sg["model"].get("test_dice") is not None:
            lines.append(f"- Segmentation model test Dice {sg['model']['test_dice']:.3f}, "
                         f"IoU {sg['model']['test_iou']:.3f} (Tool 10 test split).")
        lines.append("- A segmentation is a model output, not a radiologist's tumour outline.")
        rep["summary_text"] += "\n" + "\n".join(lines)
    ext = extended_checks(res)
    counted = [c for c in rep["reliability"]["checks"] + ext if c["counted"]]
    passed = sum(1 for c in counted if c["passed"])
    ratio = passed / len(counted) if counted else 0
    label = "HIGH" if ratio >= 0.80 else ("MODERATE" if ratio >= 0.50 else "LOW")
    # Safety override: a vote count must not call a result reliable when the image itself is in doubt.
    cap_note = None
    ood = (res.get("uncertainty") or {}).get("ood") or {}
    if ood.get("verdict") == "not_mri" and label != "LOW":
        label, cap_note = "LOW", "capped at LOW because the image does not look like a brain MRI"
    elif ood.get("verdict") == "unusual" and label == "HIGH":
        label, cap_note = "MODERATE", "capped at MODERATE because the image looks unusual for this model"
    rep["extended"] = {
        "checks": ext,
        "passed": sum(1 for c in ext if c["counted"] and c["passed"]),
        "total": sum(1 for c in ext if c["counted"]),
        "overall_passed": passed,
        "overall_total": len(counted),
        "overall_label": label,
        "cap_note": cap_note,
    }
    if any(c["counted"] for c in ext):
        lines = ["", "EXTENDED CHECKS (TOOL 8):"]
        lines += [f"- [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}" for c in ext if c["counted"]]
        lines.append(f"- Overall: {label} ({passed}/{len(counted)} checks passed)"
                     + (f", {cap_note}." if cap_note else "."))
        rep["summary_text"] += "\n" + "\n".join(lines)
    dec = decide(res)
    rep["decision"] = dec
    lines = ["", "DECISION (REJECT OPTION):"]
    if dec["status"] == "abstain":
        lines.append("- RESULT WITHHELD: the system is not reliable enough to report a class for this image.")
        lines += [f"- Reason: {r}" for r in dec["reasons"]]
        lines.append(f"- The model's best guess was '{dec['prediction']}', shown for transparency only.")
    else:
        lines.append(f"- Result reported: calibrated confidence {dec['calibrated_confidence']:.1f}% is at or above "
                     f"the reject threshold {dec['threshold']:.1f}%.")
    lines += [f"- Warning: {w}" for w in dec["warnings"]]
    if dec["source"] == "default":
        lines.append("- The threshold is a default; run Tool 8D to calibrate it on your Testing folder.")
    rep["summary_text"] += "\n" + "\n".join(lines)
    rep["ms"] = _ms(t0)
    return rep


# ================================================================
# TOOL 8 — ADVANCED ANALYSIS SUITE (nothing here retrains a model)
#   8A Uncertainty & safety : Monte Carlo dropout + out-of-distribution check
#   8B Explainability lab   : Grad-CAM++, Integrated Gradients, LIME, consensus map
#   8C Similar cases        : nearest training scans in the CNN's feature space
#   8D Model evaluation     : test-set metrics, ROC-AUC, calibration, hardest mistakes
#   8E Batch analysis       : many MRIs in one go
# ================================================================

MC_SAMPLES = 50
INDEX_PER_CLASS = 200
EVAL_BATCH = 16          # small batches keep the memory peak low on a CPU runtime
TOP_FRACTION = 0.20


def _keras():
    from tensorflow import keras
    return keras


def _fingerprint(model):
    """Short id of a model's weights, so cached results are redone after retraining."""
    import hashlib
    h = hashlib.sha1()
    ws = model.weights
    for w in (ws[0], ws[len(ws) // 2], ws[-1]):
        h.update(np.asarray(w).astype(np.float32).ravel()[:4096].tobytes())
    h.update(str(model.count_params()).encode())
    return h.hexdigest()[:12]


def _models_fingerprint():
    parts = [_fingerprint(S.effnet)]
    if S.inception is not None:
        parts.append(_fingerprint(S.inception))
    if S.densenet is not None:
        parts.append(_fingerprint(S.densenet))
    return "-".join(parts)


def _cache_path(name):
    for d in (DRIVE_FOLDER, "/content", os.getcwd()):
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            return os.path.join(d, name)
    return None


def _split_at_gap(model):
    """Feature extractor up to GlobalAveragePooling2D + the classifier head layers after it."""
    keras = _keras()
    gap_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            gap_idx = i
    if gap_idx is None:
        raise ValueError("The model has no GlobalAveragePooling2D layer.")
    head = list(model.layers[gap_idx + 1:])
    inp = model.inputs[0] if len(model.inputs) == 1 else model.inputs
    feat = keras.Model(inp, model.layers[gap_idx].output)
    return feat, head


def _head_forward(head, x, mc=False):
    keras = _keras()
    for layer in head:
        x = layer(x, training=True) if (mc and isinstance(layer, keras.layers.Dropout)) else layer(x)
    return np.asarray(x, dtype=np.float32)


_PREP = {"eff": lambda p: _prep_eff(p), "inc": lambda p: _prep_inc(p), "dense": lambda p: _prep_eff(p)}


def _model_for(key):
    return {"eff": S.effnet, "inc": S.inception, "dense": S.densenet}[key]


def _get_split(key):
    """Cached split; checks that head(features) reproduces the full model before trusting it."""
    if key in S.splits:
        return S.splits[key]
    model = _model_for(key)
    try:
        feat, head = _split_at_gap(model)
        probe = np.zeros((1,) + tuple(model.input_shape[1:]), dtype=np.float32) + 100.0
        if key == "inc":
            probe = probe / 255.0
        full = np.asarray(model(probe, training=False))
        via = _head_forward(head, np.asarray(feat(probe, training=False)))
        ok = np.allclose(full, via, atol=1e-4)
    except Exception:
        feat, head, ok = None, None, False
    S.splits[key] = (feat, head) if ok else (None, None)
    return S.splits[key]


def features(key, pils, chunk=32):
    feat, _ = _get_split(key)
    if feat is None:
        raise RuntimeError("Could not separate the feature layer of this model.")
    return _run(feat, np.stack([_PREP[key](p) for p in pils]), chunk)


# ---------------- 8A — Monte Carlo dropout + out-of-distribution ----------------

def _entropy_rows(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1)


def _mc_stats(samples):
    c = samples.shape[1]
    mean = samples.mean(0)
    pred_ent = float(_entropy_rows(mean) / np.log(c))
    alea = float(_entropy_rows(samples).mean() / np.log(c))
    epi = max(0.0, pred_ent - alea)
    votes = np.bincount(samples.argmax(1), minlength=c) / len(samples)
    top = int(np.argmax(mean))
    return {
        "prediction": S.class_names[top],
        "mean": {k: float(mean[i] * 100) for i, k in enumerate(S.class_names)},
        "std": {k: float(samples[:, i].std() * 100) for i, k in enumerate(S.class_names)},
        "votes": {k: float(votes[i] * 100) for i, k in enumerate(S.class_names)},
        "top_vote_percent": float(votes[top] * 100),
        "predictive_entropy": pred_ent,
        "aleatoric": alea,
        "epistemic": epi,
    }


def _mc_samples(key, img, T):
    model = _model_for(key)
    feat, head = _get_split(key)
    if feat is not None:
        f = np.asarray(feat(np.stack([_PREP[key](img)]), training=False))
        return _head_forward(head, np.repeat(f, T, axis=0), mc=True), "classifier-head dropout"
    x = np.repeat(np.stack([_PREP[key](img)]), T, axis=0)
    out = [np.asarray(model(x[s:s + 16], training=True)) for s in range(0, T, 16)]
    return np.concatenate(out, 0), "whole-model dropout"


def _image_checks(img):
    a = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    colour = float((np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])
                    + np.abs(a[..., 0] - a[..., 2])).mean() / (3 * 255))
    dark = float((a.mean(-1) < 20).mean())
    return {"colourfulness": colour, "dark_fraction": dark}


# ---------------- input quality validation (part of 8A) ----------------
QUALITY_SIZE = 256


def quality_stats(img):
    """Sharpness (variance of the Laplacian), contrast, brightness, size and colour of an image."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    g = np.asarray(rgb.convert("L").resize((QUALITY_SIZE, QUALITY_SIZE), Image.Resampling.BILINEAR),
                   dtype=np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    st = _image_checks(rgb)
    st.update({"width": int(w), "height": int(h), "sharpness": float(lap.var()),
               "contrast": float(g.std()), "brightness": float(g.mean())})
    return st


def _quality_reference():
    ix = S.index
    if ix is None or "q_sharp" not in ix:
        return None
    return {k: np.asarray(ix[k], dtype=np.float64) for k in ("q_sharp", "q_contrast", "q_bright")}


def validate_input(img, stats=None):
    """Input validation: is the image usable at all? Compared with the training scans when they are indexed."""
    st = stats or quality_stats(img)
    ref = _quality_reference()
    checks = []

    def add(key, name, status, value, detail):
        checks.append({"key": key, "name": name, "status": status, "value": value, "detail": detail})

    w, h = st["width"], st["height"]
    side = min(w, h)
    add("size", "Image size", "fail" if side < 96 else ("warn" if side < 180 else "ok"), f"{w} × {h} px",
        "Too small to show brain detail" if side < 96 else
        ("Smaller than typical MRI slices" if side < 180 else "Large enough"))
    ratio = max(w, h) / max(1, side)
    add("shape", "Shape", "warn" if ratio > 2.0 else "ok", f"{ratio:.2f} : 1",
        "Unusual shape for an MRI slice" if ratio > 2.0 else "Normal for an MRI slice")
    c = st["colourfulness"]
    add("colour", "Greyscale", "fail" if c > 0.06 else ("warn" if c > 0.02 else "ok"), f"{c * 100:.1f}% colour",
        "Colour image; MRI slices are greyscale" if c > 0.06 else
        ("Some colour; MRI slices are greyscale" if c > 0.02 else "Greyscale like an MRI"))
    sharp, con, bri = st["sharpness"], st["contrast"], st["brightness"]
    if con < 3:
        add("blank", "Content", "fail", f"contrast {con:.1f}", "The image is almost uniform")
    if ref is not None:
        lo = float(np.percentile(ref["q_sharp"], 1))
        hi = float(np.percentile(ref["q_sharp"], 99))
        pct = float((ref["q_sharp"] < sharp).mean() * 100)
        stt = "fail" if sharp < 0.35 * lo else ("warn" if sharp < lo or sharp > 3 * hi else "ok")
        add("sharpness", "Sharpness", stt, f"{sharp:.0f} (sharper than {pct:.0f}% of training scans)",
            "Much blurrier than every training scan" if sharp < 0.35 * lo else
            ("Blurrier than almost all training scans" if sharp < lo else
             ("Far noisier or sharper than the training scans" if sharp > 3 * hi else "Within the training range")))
        clo, chi = float(np.percentile(ref["q_contrast"], 1)), float(np.percentile(ref["q_contrast"], 99))
        add("contrast", "Contrast", "warn" if (con < 0.6 * clo or con > 1.4 * chi) else "ok", f"{con:.1f}",
            "Contrast outside the training range" if (con < 0.6 * clo or con > 1.4 * chi) else "Within the training range")
        blo, bhi = float(np.percentile(ref["q_bright"], 0.5)), float(np.percentile(ref["q_bright"], 99.5))
        add("brightness", "Brightness", "warn" if (bri < blo - 20 or bri > bhi + 20) else "ok", f"{bri:.0f} of 255",
            "Brightness outside the training range" if (bri < blo - 20 or bri > bhi + 20) else "Within the training range")
    else:
        add("sharpness", "Sharpness", "fail" if sharp < 4 else ("warn" if sharp < 12 else "ok"), f"{sharp:.0f}",
            "Very blurred" if sharp < 4 else ("Possibly blurred" if sharp < 12 else "Sharp enough (fixed threshold)"))
        add("contrast", "Contrast", "warn" if con < 12 else "ok", f"{con:.1f}",
            "Low contrast" if con < 12 else "Enough contrast (fixed threshold)")
    statuses = [c_["status"] for c_ in checks]
    verdict = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "ok")
    label = {"fail": "Unusable image", "warn": "Usable, with warnings", "ok": "Good quality"}[verdict]
    return {"verdict": verdict, "label": label, "checks": checks, "stats": st,
            "reference": "training scans" if ref is not None else "fixed thresholds (Training folder not indexed)"}


# ---------------- calibration (temperature scaling) and the reject option ----------------
REJECT_TARGET_ACCURACY = 0.98   # upper limit; the target is "halve the error rate", capped at 98% accuracy
DEFAULT_REJECT_THRESHOLD = 0.70


def reject_target(full_accuracy):
    """Accuracy the reported cases should reach: half the model's error rate, at most 98%."""
    return float(max(full_accuracy, min(REJECT_TARGET_ACCURACY, 1.0 - (1.0 - full_accuracy) / 2.0)))


def temper(p, T):
    """Temperature-scaled probabilities. log p works as logits because softmax ignores a per-row constant."""
    p = np.asarray(p, dtype=np.float64)
    if not T or abs(T - 1.0) < 1e-9:
        return p
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(P, y):
    """Find T that minimises the negative log-likelihood (golden-section search on log T)."""
    L = np.log(np.clip(np.asarray(P, dtype=np.float64), 1e-12, 1.0))
    idx = np.arange(len(y))

    def nll(log_t):
        z = L / np.exp(log_t)
        z = z - z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[idx, y]))

    a, b = np.log(0.05), np.log(20.0)
    gr = (np.sqrt(5.0) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
    return float(np.exp((a + b) / 2))


def _stratified_halves(y, seed=2024):
    rng = np.random.default_rng(seed)
    cal = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cal[idx[: len(idx) // 2]] = True
    return cal, ~cal


def _risk_coverage(conf, correct, max_points=60):
    order = np.argsort(-conf, kind="mergesort")
    cor = correct[order].astype(np.float64)
    n = np.arange(1, len(cor) + 1)
    acc = np.cumsum(cor) / n
    cov = n / len(cor)
    keep = np.unique(np.linspace(0, len(cov) - 1, min(max_points, len(cov))).round().astype(int))
    return [[float(cov[i]), float(acc[i])] for i in keep], float(np.mean(1 - acc))


def _choose_threshold(conf, correct, target):
    order = np.argsort(-conf, kind="mergesort")
    c, cor = conf[order], correct[order].astype(np.float64)
    acc = np.cumsum(cor) / np.arange(1, len(cor) + 1)
    ok = np.where(acc >= target)[0]
    if len(ok):
        return float(c[ok.max()]), True
    return float(c[len(c) // 2]), False


def reliability_settings():
    """Temperatures and the reject threshold: from Tool 8D when it has run, otherwise defaults."""
    ev = S.eval_result or {}
    cal = ev.get("calibration")
    dm = "hybrid" if S.inception is not None else "efficientnet"
    if cal and cal.get("decision"):
        d = cal["decision"]
        return {"source": "evaluation", "temperatures": {k: v["temperature"] for k, v in cal["models"].items()},
                "threshold": float(d["threshold"]), "decision_model": d["model"],
                "target_accuracy": d["target_accuracy"], "target_reached": d["target_reached"],
                "coverage": d["report"]["coverage"], "selective_accuracy": d["report"]["selective_accuracy"]}
    return {"source": "default", "temperatures": {}, "threshold": DEFAULT_REJECT_THRESHOLD, "decision_model": dm,
            "target_accuracy": REJECT_TARGET_ACCURACY, "target_reached": None, "coverage": None,
            "selective_accuracy": None}


def decide(res):
    """Reject option: withhold the class when the input or the calibrated confidence is not reliable enough."""
    rs = reliability_settings()
    hyb = res.get("hybrid") or {}
    cal = hyb.get("calibrated") or {}
    conf = float(cal.get("confidence", (hyb.get("hybrid") or {}).get("confidence", 0.0))) / 100.0
    unc = res.get("uncertainty") or {}
    ood = unc.get("ood") or {}
    q = unc.get("quality") or {}
    reasons, warnings = [], []
    if ood.get("verdict") == "not_mri":
        reasons.append("The image does not look like a brain MRI.")
    elif ood.get("verdict") == "unusual":
        warnings.append("The image is unusual compared with the training scans.")
    for c in q.get("checks", []):
        if c["status"] == "fail":
            reasons.append(f"{c['name']}: {c['detail']}.")
        elif c["status"] == "warn":
            warnings.append(f"{c['name']}: {c['detail']}.")
    sg = res.get("segment") or {}
    if sg.get("available") and sg.get("consistent_with_classifier") is False:
        warnings.append("The U-Net segmentation and the classifier disagree about whether a tumour is present.")
    if conf < rs["threshold"]:
        reasons.append(f"Calibrated confidence {conf * 100:.1f}% is below the reject threshold "
                       f"{rs['threshold'] * 100:.1f}%.")
    return {"status": "abstain" if reasons else "report", "reasons": reasons, "warnings": warnings,
            "calibrated_confidence": conf * 100, "threshold": rs["threshold"] * 100,
            "temperature": cal.get("temperature", 1.0), "source": rs["source"],
            "prediction": cal.get("prediction") or (hyb.get("hybrid") or {}).get("prediction"),
            "checked_input": bool(unc)}


# ---------------- speed and size of each model ----------------

def benchmark_models(models=None, runs=10, batch=8):
    out = {}
    for key, label, model, size, scale in (models or _eval_models()):
        x1 = np.random.default_rng(0).uniform(0, 255, (1, size[0], size[1], 3)).astype(np.float32) * scale
        xb = np.repeat(x1, batch, 0)
        with S.lock:
            for _ in range(2):
                model(x1, training=False)
            t = []
            for _ in range(runs):
                t0 = time.perf_counter()
                model(x1, training=False)
                t.append(time.perf_counter() - t0)
            model(xb, training=False)
            t0 = time.perf_counter()
            for _ in range(2):
                model(xb, training=False)
            thr = 2 * batch / (time.perf_counter() - t0)
        params = int(model.count_params())
        path = S.model_paths.get(key)
        out[key] = {"label": label, "parameters": params, "weights_mb": params * 4 / 1e6,
                    "file_mb": (os.path.getsize(path) / 1e6) if path and os.path.exists(path) else None,
                    "latency_ms": float(np.median(t) * 1000), "throughput": float(thr),
                    "input": f"{size[0]} x {size[1]}"}
    if "efficientnet" in out and "inception" in out:
        out["hybrid"] = {"label": "Hybrid vote (Tool 3)",
                         "parameters": out["efficientnet"]["parameters"] + out["inception"]["parameters"],
                         "weights_mb": out["efficientnet"]["weights_mb"] + out["inception"]["weights_mb"],
                         "file_mb": None,
                         "latency_ms": out["efficientnet"]["latency_ms"] + out["inception"]["latency_ms"],
                         "throughput": 1.0 / (1.0 / out["efficientnet"]["throughput"] + 1.0 / out["inception"]["throughput"]),
                         "input": "both"}
    return out


def _fit_gauss(feats, labels, k=32):
    """Class-conditional Gaussian (shared covariance) in a whitened PCA space of the CNN features."""
    mu = feats.mean(0)
    X = feats - mu
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    k = int(min(k, vt.shape[0], len(X) - 1))
    comps = vt[:k]
    scale = sv[:k] / np.sqrt(max(1, len(X) - 1)) + 1e-6
    Z = (X @ comps.T) / scale
    means = np.stack([Z[labels == c].mean(0) if (labels == c).any() else np.zeros(k)
                      for c in range(len(S.class_names))])
    resid = Z - means[labels]
    cov = resid.T @ resid / len(Z) + 0.2 * np.eye(k)
    return {"mu": mu, "comps": comps, "scale": scale, "means": means, "prec": np.linalg.inv(cov)}


def _gauss_distance(g, F):
    Z = ((F - g["mu"]) @ g["comps"].T) / g["scale"]
    return _maha(Z, g["means"], g["prec"])


def _fit_ood(feats, labels, seed=3):
    # reference distances come from scans the model of "normal" did not see (2-fold cross-fitting),
    # otherwise ordinary new MRIs would look further away than the training scans and get flagged
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(feats))
    d_ref = np.zeros(len(feats))
    for f in (0, 1):
        g = _fit_gauss(feats[fold != f], labels[fold != f])
        d_ref[fold == f] = _gauss_distance(g, feats[fold == f])
    g = _fit_gauss(feats, labels)
    return {"mu": g["mu"].astype(np.float32), "comps": g["comps"].astype(np.float32),
            "scale": g["scale"].astype(np.float32), "means": g["means"].astype(np.float32),
            "prec": g["prec"].astype(np.float32), "d_train": np.sort(d_ref).astype(np.float32)}


def _maha(Z, means, prec):
    d = np.stack([np.einsum("ij,jk,ik->i", Z - m, prec, Z - m) for m in means], axis=1)
    return np.sqrt(np.maximum(d, 0)).min(1)


def _ood_distance(f):
    ix = S.index
    d = float(_gauss_distance(ix, f[None, :])[0])
    dt = ix["d_train"]
    p99 = float(np.percentile(dt, 99))
    pct = float((dt < d).mean() * 100)
    return d, p99, pct


def _wait_for_index(timeout=150):
    t_end = time.time() + timeout
    while S.index_status.get("state") == "building" and time.time() < t_end:
        time.sleep(0.5)
    return S.index is not None


def stage_uncertainty(case):
    t0 = time.perf_counter()
    if "hybrid_probs" not in case:
        with S.lock:
            case["results"]["hybrid"] = stage_hybrid(case)
    img = case["image"]
    _wait_for_index()
    with S.lock:
        eff_s, method = _mc_samples("eff", img, MC_SAMPLES)
        models = {"efficientnet": _mc_stats(eff_s)}
        hyb_s = eff_s
        if S.inception is not None:
            inc_s, _ = _mc_samples("inc", img, MC_SAMPLES)
            models["inception"] = _mc_stats(inc_s)
            hyb_s = 0.5 * (eff_s + inc_s)
        hybrid = _mc_stats(hyb_s)
        f = None
        if S.index is not None:
            f = features("eff", [img])[0]
            case["eff_feat"] = f

    epi = hybrid["epistemic"]
    level = "Low" if epi < 0.03 else ("Moderate" if epi < 0.10 else "High")
    det = S.class_names[int(np.argmax(case["hybrid_probs"]))]

    quality = validate_input(img)
    checks = _image_checks(img)
    feature_check = None
    if f is not None:
        d, p99, pct = _ood_distance(f)
        feature_check = {"distance": d, "p99": p99, "ratio": d / p99 if p99 > 0 else 0.0,
                         "percentile": pct, "reference_scans": int(len(S.index["labels"]))}
    max_soft = float(np.max(case["hybrid_probs"]) * 100)
    ratio = feature_check["ratio"] if feature_check else 0.0
    reasons = []
    if checks["colourfulness"] > 0.06:
        reasons.append("The image is in colour; MRI slices are greyscale.")
    elif checks["colourfulness"] > 0.02:
        reasons.append("The image has some colour, which MRI slices do not.")
    if feature_check and ratio > 2.5:
        reasons.append("Its features are far outside every training scan.")
    elif feature_check and ratio > 1.25:
        reasons.append("Its features are further from the training scans than almost all of them.")
    # Low confidence is reported by Tool 3D and the uncertainty panel; it does not make the image itself unusual.
    if checks["colourfulness"] > 0.06 or ratio > 2.5:
        verdict, label = "not_mri", "Probably not a brain MRI"
    elif reasons:
        verdict, label = "unusual", "Unusual image — check it"
    else:
        verdict, label = "ok", "Looks like a brain MRI"

    return {
        "method": method,
        "samples": MC_SAMPLES,
        "deterministic_prediction": det,
        "hybrid": hybrid,
        "models": models,
        "epistemic_level": level,
        "mc_agrees": hybrid["prediction"] == det,
        "ood": {"verdict": verdict, "label": label, "reasons": reasons, "image": checks,
                "feature_check": feature_check, "max_probability": max_soft},
        "quality": quality,
        "ms": _ms(t0),
    }


# ---------------- index of training scans (8A feature check + 8C similar cases) ----------------

def _list_train(per_class, seed=11):
    rng = random.Random(seed)
    items = []
    for ci, cls in enumerate(S.class_names):
        folder = os.path.join(S.train_dir, cls)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXT))
        for f in rng.sample(files, min(per_class, len(files))):
            items.append((os.path.join(cls, f), ci))
    return items


def build_index(per_class=INDEX_PER_CLASS, verbose=False, force=False):
    """Features of a sample of training scans. Cached to Drive, so it is built once per trained model."""
    if S.train_dir is None:
        S.index_status = {"state": "unavailable",
                          "message": "Training folder not found. Run the dataset and unzip cells first."}
        return None
    cache = _cache_path(f"neuronova_index_v3_{_fingerprint(S.effnet)}_{per_class}.npz")
    if cache and os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=False)
        S.index = {k: z[k] for k in z.files}
        S.index["paths"] = [str(p) for p in S.index["paths"]]
        S.index_status = {"state": "ready", "done": len(S.index["paths"]), "total": len(S.index["paths"]),
                          "cached": True}
        if verbose:
            print(f"  Training-scan index loaded from cache ({len(S.index['paths'])} scans)")
        return S.index
    items = _list_train(per_class)
    S.index_status = {"state": "building", "done": 0, "total": len(items)}
    feats, qual = [], []
    for s in range(0, len(items), 32):
        chunk = items[s:s + 32]
        pils = [Image.open(os.path.join(S.train_dir, p)).convert("RGB") for p, _ in chunk]
        qual += [quality_stats(im) for im in pils]
        with S.lock:
            feats.append(features("eff", pils))
        S.index_status["done"] = min(len(items), s + 32)
        if verbose and (s // 32) % 5 == 0:
            print(f"  indexing training scans {S.index_status['done']}/{len(items)}")
    feats = np.concatenate(feats, 0).astype(np.float32)
    labels = np.array([c for _, c in items], dtype=np.int64)
    ood = _fit_ood(feats, labels)
    centred = feats - ood["mu"]
    norm = centred / (np.linalg.norm(centred, axis=1, keepdims=True) + 1e-8)
    idx = {"norm": norm.astype(np.float16), "labels": labels, "paths": np.array([p for p, _ in items]), **ood,
           "q_sharp": np.array([q["sharpness"] for q in qual], dtype=np.float32),
           "q_contrast": np.array([q["contrast"] for q in qual], dtype=np.float32),
           "q_bright": np.array([q["brightness"] for q in qual], dtype=np.float32)}
    if cache:
        try:
            np.savez_compressed(cache, **idx)
        except Exception:
            pass
    idx["paths"] = [str(p) for p in idx["paths"]]
    S.index = idx
    S.index_status = {"state": "ready", "done": len(items), "total": len(items), "cached": False}
    if verbose:
        print(f"  Training-scan index ready ({len(items)} scans)")
    return S.index


def _start_index_thread():
    if S.train_dir is None:
        S.index_status = {"state": "unavailable",
                          "message": "Training folder not found. Run the dataset and unzip cells first."}
        return

    def work():
        try:
            build_index()
        except Exception as exc:
            S.index_status = {"state": "error", "message": f"{type(exc).__name__}: {exc}"}
    S.index_status = {"state": "building", "done": 0, "total": 0}
    threading.Thread(target=work, daemon=True).start()


def wait_for_index(verbose=True, timeout=3600):
    """Notebook helper: wait for the similar-case index (built in the background by setup/launch)."""
    if S.index is None and S.index_status.get("state") in (None, "idle"):
        _start_index_thread()
    last, t_end = None, time.time() + timeout
    while S.index_status.get("state") == "building" and time.time() < t_end:
        st = S.index_status
        msg = f"  Indexing training scans {st.get('done', 0)}/{st.get('total', 0) or '...'}"
        if verbose and msg != last:
            print(msg, end="\r", flush=True)
            last = msg
        time.sleep(1)
    if verbose:
        if S.index is not None:
            how = "loaded from your Drive folder" if S.index_status.get("cached") else "built and saved to your Drive folder"
            print(f"  Similar-case index ready: {len(S.index['labels'])} training scans ({how})        ")
        else:
            print("  Similar-case index not available:", S.index_status.get("message") or S.index_status.get("state"))
    return None


def _is_duplicate(img, path):
    """Same picture as a training scan? Compare small greyscale versions pixel by pixel."""
    try:
        a = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
        b = np.asarray(Image.open(path).convert("L").resize((64, 64)), dtype=np.float32)
        diff = np.abs(a - b)
        return bool(diff.mean() < 1.0 and np.percentile(diff, 99) < 8)
    except Exception:
        return False


def _thumb(path, size=160):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    return _png_data_url(im, "JPEG")


# ---------------- 8C — similar cases ----------------

def stage_similar(case, k=5, vote_k=10):
    t0 = time.perf_counter()
    if not _wait_for_index() or S.index is None:
        msg = S.index_status.get("message") or "The training-scan index is not available."
        return {"available": False, "message": msg, "ms": _ms(t0)}
    if "hybrid_probs" not in case:
        with S.lock:
            case["results"]["hybrid"] = stage_hybrid(case)
    f = case.get("eff_feat")
    if f is None:
        with S.lock:
            f = features("eff", [case["image"]])[0]
        case["eff_feat"] = f
    fc = f - S.index["mu"]
    q = fc / (np.linalg.norm(fc) + 1e-8)
    sims = S.index["norm"].astype(np.float32) @ q
    order = np.argsort(-sims)[:max(k, vote_k)]
    pred = S.class_names[int(np.argmax(case["hybrid_probs"]))]
    neighbours = []
    for i in order[:k]:
        rel = S.index["paths"][i]
        neighbours.append({"label": S.class_names[int(S.index["labels"][i])], "similarity": float(sims[i] * 100),
                           "file": os.path.basename(rel), "thumb": _thumb(os.path.join(S.train_dir, rel))})
    votes = {c: 0 for c in S.class_names}
    for i in order[:vote_k]:
        votes[S.class_names[int(S.index["labels"][i])]] += 1
    agreement = votes[pred] / float(vote_k)
    verdict = (((case["results"].get("uncertainty") or {}).get("ood")) or {}).get("verdict")
    warning = None
    if verdict == "not_mri":
        warning = ("This image does not look like a brain MRI, so these are only the least different training scans, "
                   "not truly similar ones. Their labels say nothing about this image.")
    elif verdict == "unusual":
        warning = "The safety check found this image unusual for the model, so read these matches with care."
    return {
        "available": True,
        "warning": warning,
        "neighbours": neighbours,
        "votes": votes,
        "vote_k": vote_k,
        "prediction": pred,
        "agreement": agreement,
        "duplicate": _is_duplicate(case["image"], os.path.join(S.train_dir, S.index["paths"][order[0]]))
        if sims[order[0]] > 0.95 else False,
        "reference_scans": int(len(S.index["labels"])),
        "ms": _ms(t0),
    }


# ---------------- 8B — explainability lab ----------------

def _norm_map(m):
    m = np.maximum(np.asarray(m, dtype=np.float32), 0)
    hi = float(np.percentile(m, 99.5)) if m.max() > 0 else 0.0
    return np.clip(m / hi, 0, 1) if hi > 0 else m


def _resize_map(m, size=224):
    return np.array(Image.fromarray(np.uint8(np.clip(m, 0, 1) * 255)).resize(
        (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _blur_map(m, radius=4):
    from PIL import ImageFilter
    im = Image.fromarray(np.uint8(np.clip(m, 0, 1) * 255)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, dtype=np.float32) / 255.0


def _top_mask(m, frac=TOP_FRACTION):
    thr = np.quantile(m, 1 - frac)
    return m >= max(thr, 1e-6)


def gradcam_plus_plus(x, idx):
    import tensorflow as tf
    _ensure_grad_model()
    with tf.GradientTape() as tape:
        conv, preds = S.grad_model(x, training=False)
        score = preds[:, idx]
    g = tape.gradient(score, conv)
    g2, g3 = g ** 2, g ** 3
    sum_a = tf.reduce_sum(conv, axis=(1, 2), keepdims=True)
    alpha = g2 / (2.0 * g2 + sum_a * g3 + 1e-8)
    w = tf.reduce_sum(alpha * tf.nn.relu(g), axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * w[0][tf.newaxis, tf.newaxis, :], axis=-1)).numpy()
    return _resize_map(_norm_map(cam))


def integrated_gradients(x, idx, steps=32):
    import tensorflow as tf
    base = np.zeros_like(x)
    alphas = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
    total = np.zeros_like(x[0])
    for s in range(0, steps, 16):
        a = alphas[s:s + 16][:, None, None, None]
        xs = tf.convert_to_tensor(base + a * (x - base))
        with tf.GradientTape() as tape:
            tape.watch(xs)
            out = S.effnet(xs, training=False)[:, idx]
        total += tape.gradient(out, xs).numpy().sum(0)
    attr = (x[0] - base[0]) * total / steps
    return _norm_map(_blur_map(_norm_map(np.abs(attr).sum(-1)), radius=3))


def lime_map(img224, idx, n_samples=300, seed=0):
    arr = np.asarray(img224, dtype=np.float32)
    try:
        from skimage.segmentation import slic
        seg = slic(arr / 255.0, n_segments=50, compactness=10, start_label=0, channel_axis=-1)
    except Exception:
        yy, xx = np.mgrid[0:224, 0:224]
        seg = (yy // 32) * 7 + (xx // 32)
    nseg = int(seg.max()) + 1
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, size=(n_samples, nseg)).astype(np.float32)
    Z[0] = 1
    fill = np.full_like(arr, arr.mean())
    ys = []
    for s in range(0, n_samples, 32):
        batch = []
        for z in Z[s:s + 32]:
            off = (z[seg] == 0)
            im = arr.copy()
            im[off] = fill[off]
            batch.append(im)
        ys.append(np.asarray(S.effnet(np.stack(batch), training=False))[:, idx])
    y = np.concatenate(ys)
    d = 1 - Z.sum(1) / (np.sqrt(np.maximum(Z.sum(1), 1)) * np.sqrt(nseg))
    w = np.exp(-(d ** 2) / 0.25 ** 2)
    X = np.hstack([Z, np.ones((n_samples, 1), dtype=np.float32)])
    A = X.T @ (X * w[:, None]) + 1.0 * np.eye(nseg + 1)
    sol = np.linalg.solve(A, X.T @ (w * y))
    coef = sol[:nseg]
    m = np.maximum(coef, 0)[seg]
    top = [int(i) for i in np.argsort(-coef)[:5]]
    fit = X @ sol
    # When hiding segments never changes the class score (a saturated prediction), the fit quality is undefined.
    r2 = float(np.corrcoef(fit, y)[0, 1]) if y.std() > 1e-6 and fit.std() > 1e-9 else None
    if r2 is not None and not np.isfinite(r2):
        r2 = None
    return _norm_map(m), {"segments": nseg, "samples": n_samples, "top_segments": top, "r2_proxy": r2,
                          "score_range": [float(y.min()), float(y.max())]}


def stage_xai(case):
    t0 = time.perf_counter()
    if "classify" not in case["results"]:
        with S.lock:
            case["results"]["classify"] = stage_classify(case)
    idx = case["results"]["classify"]["index"]
    img = case["image"]
    img224 = img.convert("RGB").resize((224, 224))
    x = np.expand_dims(np.array(img224, dtype=np.float32), 0)
    timings = {}
    with S.lock:
        if "heatmap_resized" not in case:
            case["results"]["gradcam"] = stage_gradcam(case)
        maps = {"gradcam": _norm_map(_resize_map(case["heatmap_resized"]))}
        t = time.perf_counter(); maps["gradcam_pp"] = gradcam_plus_plus(x, idx); timings["gradcam_pp"] = _ms(t)
        t = time.perf_counter(); maps["integrated_gradients"] = integrated_gradients(x, idx); timings["integrated_gradients"] = _ms(t)
        t = time.perf_counter(); maps["lime"], lime_info = lime_map(img224, idx); timings["lime"] = _ms(t)
    names = ["gradcam", "gradcam_pp", "integrated_gradients", "lime"]
    masks = {k: _top_mask(maps[k]) for k in names}
    iou, corr = {}, {}
    for a in names:
        for b in names:
            inter = np.logical_and(masks[a], masks[b]).sum()
            union = np.logical_or(masks[a], masks[b]).sum()
            iou[f"{a}|{b}"] = float(inter / union) if union else 0.0
            ma, mb = maps[a].ravel(), maps[b].ravel()
            corr[f"{a}|{b}"] = float(np.corrcoef(ma, mb)[0, 1]) if ma.std() > 0 and mb.std() > 0 else 0.0
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    mean_iou = float(np.mean([iou[f"{a}|{b}"] for a, b in pairs]))
    consensus = _norm_map(np.mean([maps[k] for k in names], axis=0))
    try:
        t = time.perf_counter()
        faith = deletion_curves(x, idx, {**{k: maps[k] for k in names}, "consensus": consensus})
        timings["deletion_test"] = _ms(t)
    except Exception as exc:
        faith = {"error": f"{type(exc).__name__}: {exc}"}
    cm = _top_mask(consensus)
    ys, xs = np.where(cm)
    base = np.array(img224, dtype=np.uint8)
    if mean_iou >= 0.45:
        strength = "Strong consensus"
    elif mean_iou >= 0.30:
        strength = "Moderate consensus"
    else:
        strength = "Weak consensus"
    return {
        "methods": names + ["consensus"],
        "overlays": {k: _png_data_url(_overlay(base, v)) for k, v in list(maps.items()) + [("consensus", consensus)]},
        "iou": iou,
        "correlation": corr,
        "mean_pairwise_iou": mean_iou,
        "strength": strength,
        "consensus_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
        "lime": lime_info,
        "faithfulness": faith,
        "timings_ms": timings,
        "ms": _ms(t0),
    }


# ---------------- extended reliability checks (Tool 3D's four + Tool 8's four) ----------------

def extended_checks(res):
    unc, sim, xai = res.get("uncertainty"), res.get("similar"), res.get("xai")
    checks = [
        {"name": "Image looks like a brain MRI (out-of-distribution check)",
         "passed": bool(unc and unc["ood"]["verdict"] == "ok"), "counted": bool(unc)},
        {"name": "Low epistemic uncertainty (Monte Carlo dropout)",
         "passed": bool(unc and unc["epistemic_level"] == "Low"), "counted": bool(unc)},
        {"name": "Most of the 10 most similar training scans share the predicted class",
         "passed": bool(sim and sim.get("available") and sim["agreement"] >= 0.6),
         "counted": bool(sim and sim.get("available"))},
        {"name": "Four explanation methods point at the same area (mean overlap at least 0.30)",
         "passed": bool(xai and xai["mean_pairwise_iou"] >= 0.30), "counted": bool(xai)},
        {"name": "Image passed the quality checks (size, sharpness, contrast, colour)",
         "passed": bool(unc and unc.get("quality") and unc["quality"]["verdict"] != "fail"),
         "counted": bool(unc and unc.get("quality"))},
    ]
    sg = res.get("segment") or {}
    if sg.get("available") and sg.get("consistent_with_classifier") is not None:
        checks.append({"name": "U-Net segmentation agrees with the classifier (a region only for a tumour class)",
                       "passed": bool(sg["consistent_with_classifier"]), "counted": True})
    hyb = res.get("hybrid") or {}
    if hyb.get("calibrated"):
        cal = hyb["calibrated"]
        checks.append({"name": "Calibrated confidence is above the reject threshold",
                       "passed": bool(cal["confidence"] >= cal["threshold"]), "counted": True})
    return checks


# ---------------- 8D — model evaluation on the Testing folder ----------------

def _roc_points(scores, positives, max_points=60):
    order = np.argsort(-scores, kind="mergesort")
    pos = positives[order].astype(np.float64)
    tps = np.cumsum(pos)
    fps = np.cumsum(1 - pos)
    P, N = max(1.0, pos.sum()), max(1.0, (1 - pos).sum())
    tpr = np.concatenate([[0], tps / P])
    fpr = np.concatenate([[0], fps / N])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).round().astype(int))
    return auc, [[float(fpr[i]), float(tpr[i])] for i in keep]


def _pr_points(scores, positives, max_points=60):
    order = np.argsort(-scores, kind="mergesort")
    pos = positives[order].astype(np.float64)
    tp = np.cumsum(pos)
    fp = np.cumsum(1 - pos)
    npos = max(1.0, pos.sum())
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / npos
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    keep = np.unique(np.linspace(0, len(rec) - 1, min(max_points, len(rec))).round().astype(int))
    return ap, [[0.0, 1.0]] + [[float(rec[i]), float(prec[i])] for i in keep]


def compute_metrics(y, P, class_names):
    C = P.shape[1]
    N = len(y)
    pred = P.argmax(1)
    conf = P.max(1)
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(np.float64)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(C), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(C), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
    support = cm.sum(1)
    acc = float(tp.sum() / N)
    pe = float((cm.sum(0) * cm.sum(1)).sum() / N ** 2)
    kappa = float((acc - pe) / (1 - pe)) if pe < 1 else 0.0
    s, c = float(N), float(tp.sum())
    pk, tk = cm.sum(0).astype(np.float64), cm.sum(1).astype(np.float64)
    den = np.sqrt((s ** 2 - (pk ** 2).sum()) * (s ** 2 - (tk ** 2).sum()))
    mcc = float((c * s - (pk * tk).sum()) / den) if den > 0 else 0.0
    per_class, aucs, rocs, aps, prs = [], [], {}, [], {}
    for k in range(C):
        auc, pts = _roc_points(P[:, k], (y == k))
        ap, ppts = _pr_points(P[:, k], (y == k))
        aucs.append(auc)
        aps.append(ap)
        rocs[class_names[k]] = pts
        prs[class_names[k]] = ppts
        per_class.append({"class": class_names[k], "precision": float(prec[k]), "recall": float(rec[k]),
                          "f1": float(f1[k]), "auc": auc, "ap": ap, "support": int(support[k])})
    bins = np.linspace(0, 1, 11)
    calib, ece = [], 0.0
    for b in range(10):
        m = (conf > bins[b]) & (conf <= bins[b + 1]) if b else (conf >= 0) & (conf <= bins[1])
        n = int(m.sum())
        if n:
            a_b, c_b = float((pred[m] == y[m]).mean()), float(conf[m].mean())
            ece += n / N * abs(a_b - c_b)
            calib.append({"low": float(bins[b]), "high": float(bins[b + 1]), "count": n,
                          "accuracy": a_b, "confidence": c_b})
    onehot = np.eye(C)[y]
    top2 = float(np.mean([y[i] in np.argsort(-P[i])[:2] for i in range(N)]))
    return {
        "n": int(N),
        "accuracy": acc,
        "top2_accuracy": top2,
        "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()), "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "macro_auc": float(np.mean(aucs)),
        "macro_ap": float(np.mean(aps)),
        "kappa": kappa, "mcc": mcc,
        "ece": float(ece),
        "brier": float(((P - onehot) ** 2).sum(1).mean()),
        "log_loss": float(-np.log(np.clip(P[np.arange(N), y], 1e-12, 1)).mean()),
        "confusion": cm.tolist(),
        "per_class": per_class,
        "roc": rocs,
        "pr": prs,
        "calibration": calib,
    }


def _eval_models():
    out = [("efficientnet", "EfficientNetV2-B0", S.effnet, (224, 224), 1.0)]
    if S.inception is not None:
        out.append(("inception", "InceptionV3", S.inception, (299, 299), 1.0 / 255.0))
    if S.densenet is not None:
        out.append(("densenet", "DenseNet121", S.densenet, (224, 224), 1.0))
    return out


def run_evaluation(verbose=False, force=False):
    """Run every model over the Testing folder (same loading as your training cells) and score them."""
    import tensorflow as tf
    if S.test_dir is None:
        raise RuntimeError("Testing folder not found. Run the dataset and unzip cells first.")
    fp = _models_fingerprint()
    cache = _cache_path(f"neuronova_evaluation_v3_{fp}.json")
    if cache and os.path.exists(cache) and not force:
        with open(cache, "r", encoding="utf-8") as fh:
            res = json.load(fh)
        S.eval_result = res
        S.eval_status = {"state": "done", "cached": True}
        if verbose:
            print("  Evaluation loaded from cache:", cache)
        return res
    t0 = time.time()
    models = _eval_models()
    S.eval_status = {"state": "running", "done": 0, "total": 0, "phase": "loading Testing folder"}
    probs, y, paths, ds_classes = {}, None, None, None
    def load(size):
        kw = dict(labels="inferred", label_mode="int", image_size=size, batch_size=EVAL_BATCH, shuffle=False)
        try:
            return tf.keras.utils.image_dataset_from_directory(S.test_dir, verbose=False, **kw)
        except TypeError:
            return tf.keras.utils.image_dataset_from_directory(S.test_dir, **kw)

    for key, label, model, size, scale in models:
        ds = load(size)
        if y is None:
            paths = list(ds.file_paths)
            ds_classes = list(ds.class_names)
            S.eval_status["total"] = len(paths) * len(models)
        outs, labels = [], []
        for xb, yb in ds:
            with S.lock:
                outs.append(np.asarray(model(xb * scale, training=False), dtype=np.float64))
            labels.append(yb.numpy())
            S.eval_status["done"] += len(yb)
            S.eval_status["phase"] = f"{label}: {sum(len(o) for o in outs)}/{len(paths)} images"
            if verbose and len(outs) % 10 == 0:
                print(f"  {S.eval_status['phase']}")
        probs[key] = np.concatenate(outs, 0)
        y = np.concatenate(labels, 0)
    if ds_classes != list(S.class_names):
        order = [ds_classes.index(c) for c in S.class_names]
        remap = {old: new for new, old in enumerate(order)}
        y = np.array([remap[v] for v in y])
        probs = {k: v[:, order] for k, v in probs.items()}
    names = {k: lab for k, lab, *_ in models}
    if "inception" in probs:
        probs["hybrid"] = 0.5 * probs["efficientnet"] + 0.5 * probs["inception"]
        names["hybrid"] = "Hybrid vote (Tool 3)"
    if "densenet" in probs and "inception" in probs:
        probs["ensemble3"] = (probs["efficientnet"] + probs["inception"] + probs["densenet"]) / 3.0
        names["ensemble3"] = "3-model ensemble"
    metrics = {k: {"label": names[k], **compute_metrics(y, P, S.class_names)} for k, P in probs.items()}
    best_key = "hybrid" if "hybrid" in probs else "efficientnet"
    P = probs[best_key]
    pred = P.argmax(1)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-P[wrong].max(1))][:8]
    mistakes = []
    for i in wrong:
        try:
            th = _thumb(paths[i], 128)
        except Exception:
            th = None
        mistakes.append({"file": os.path.basename(paths[i]), "true": S.class_names[int(y[i])],
                         "predicted": S.class_names[int(pred[i])], "confidence": float(P[i].max() * 100),
                         "thumb": th})
    # calibration and reject option: fit on one half of Testing, report on the other half
    S.eval_status["phase"] = "calibrating (temperature scaling)"
    cal_mask, rep_mask = _stratified_halves(y)
    calibration = {"split": {"calibration": int(cal_mask.sum()), "report": int(rep_mask.sum()), "seed": 2024},
                   "models": {}}
    for k, Pk in probs.items():
        T = fit_temperature(Pk[cal_mask], y[cal_mask])
        before = compute_metrics(y[rep_mask], Pk[rep_mask], S.class_names)
        after = compute_metrics(y[rep_mask], temper(Pk[rep_mask], T), S.class_names)
        pick = lambda m: {"ece": m["ece"], "brier": m["brier"], "log_loss": m["log_loss"], "accuracy": m["accuracy"]}
        calibration["models"][k] = {"label": names[k], "temperature": T, "before": pick(before), "after": pick(after),
                                    "reliability_before": before["calibration"],
                                    "reliability_after": after["calibration"]}
    dk = "hybrid" if "hybrid" in probs else "efficientnet"
    Td = calibration["models"][dk]["temperature"]
    cc = temper(probs[dk][cal_mask], Td)
    target = reject_target(float((cc.argmax(1) == y[cal_mask]).mean()))
    thr, reached = _choose_threshold(cc.max(1), cc.argmax(1) == y[cal_mask], target)
    cr = temper(probs[dk][rep_mask], Td)
    conf_r, cor_r = cr.max(1), cr.argmax(1) == y[rep_mask]
    keep = conf_r >= thr
    rc, aurc = _risk_coverage(conf_r, cor_r)
    calibration["decision"] = {
        "model": dk, "label": names[dk], "temperature": Td, "target_accuracy": target,
        "target_reached": bool(reached), "threshold": thr,
        "report": {"n": int(rep_mask.sum()), "coverage": float(keep.mean()),
                   "selective_accuracy": float(cor_r[keep].mean()) if keep.any() else None,
                   "full_accuracy": float(cor_r.mean()), "rejected": int((~keep).sum()),
                   "rejected_accuracy": float(cor_r[~keep].mean()) if (~keep).any() else None},
        "risk_coverage": rc, "aurc": aurc,
    }
    S.eval_status["phase"] = "measuring speed and size"
    try:
        bench = benchmark_models(models)
    except Exception as exc:
        bench = {"error": f"{type(exc).__name__}: {exc}"}
    res = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "n_images": int(len(y)),
        "calibration": calibration,
        "benchmark": bench,
        "classes": list(S.class_names),
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(S.class_names)},
        "models": metrics,
        "order": list(probs.keys()),
        "mistakes_model": names[best_key],
        "mistakes": mistakes,
        "fingerprint": fp,
        "note": ("The Testing folder was also used for early stopping during training, "
                 "so these scores are likely a little optimistic."),
    }
    if cache:
        try:
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(res, fh)
        except Exception:
            pass
    S.eval_result = res
    S.eval_status = {"state": "done", "cached": False}
    import gc
    gc.collect()
    if verbose:
        print(f"  Evaluation finished in {res['seconds']} s")
    return res


def _start_eval_thread(force=False):
    if S.eval_status.get("state") == "running":
        return

    def work():
        try:
            run_evaluation(force=force)
        except Exception as exc:
            S.eval_status = {"state": "error", "message": f"{type(exc).__name__}: {exc}"}
    S.eval_status = {"state": "running", "done": 0, "total": 0, "phase": "starting"}
    threading.Thread(target=work, daemon=True).start()


def evaluate(verbose=True, force=False):
    """Notebook helper for Tool 8D: score every model on the Testing folder, with a progress line.
    The result is saved to your Drive folder, so later runs load it instantly (force=True re-runs it)."""
    if S.test_dir is None:
        raise RuntimeError("Testing folder not found. Run the dataset and unzip cells first.")
    if S.eval_status.get("state") == "done" and S.eval_result and not force:
        if verbose:
            print(f"  Evaluation loaded (scored {S.eval_result['n_images']} Testing images on "
                  f"{S.eval_result['generated_at'][:16].replace('T', ' ')} UTC). Use force=True to run it again.")
        return S.eval_result
    if S.eval_status.get("state") != "running":
        _start_eval_thread(force=force)
    last = None
    while S.eval_status.get("state") == "running":
        st = S.eval_status
        k = st["done"] / st["total"] if st.get("total") else 0.0
        msg = f"  {st.get('phase') or 'starting'} · {k * 100:.0f}%"
        if verbose and msg != last:
            print(msg.ljust(70), end="\r", flush=True)
            last = msg
        time.sleep(1)
    if S.eval_status.get("state") == "error":
        raise RuntimeError("Evaluation failed: " + S.eval_status.get("message", ""))
    if verbose and S.eval_result:
        print(f"  Evaluation finished: {S.eval_result['n_images']} Testing images in "
              f"{S.eval_result['seconds']} s (saved to your Drive folder)".ljust(70))
    return S.eval_result


def _load_cached_evaluation():
    try:
        cache = _cache_path(f"neuronova_evaluation_v3_{_models_fingerprint()}.json")
        if cache and os.path.exists(cache):
            with open(cache, "r", encoding="utf-8") as fh:
                S.eval_result = json.load(fh)
            S.eval_status = {"state": "done", "cached": True}
    except Exception:
        pass


# ---------------- 8E — batch analysis ----------------

def batch_predict(images, names=None):
    """images: list of PIL images or file paths. Returns one row per image."""
    rows = []
    for i, im in enumerate(images):
        name = names[i] if names else (os.path.basename(im) if isinstance(im, str) else f"image_{i + 1}")
        try:
            pil = Image.open(im).convert("RGB") if isinstance(im, str) else im.convert("RGB")
        except Exception as exc:
            rows.append({"file": name, "error": f"Could not open: {exc}"})
            continue
        with S.lock:
            hyb, eff, inc = predict_hybrid([pil])
        h, e = hyb[0], eff[0]
        hi, ei = int(np.argmax(h)), int(np.argmax(e))
        colour = _image_checks(pil)["colourfulness"]
        ratio = None
        if S.index is not None:
            with S.lock:
                f = features("eff", [pil])[0]
            d, p99, _ = _ood_distance(f)
            ratio = d / p99 if p99 > 0 else None
        if colour > 0.06 or (ratio is not None and ratio > 2.5):
            safety = "Probably not a brain MRI"
        elif colour > 0.02 or (ratio is not None and ratio > 1.25):
            safety = "Unusual — check it"
        else:
            safety = "Looks like a brain MRI"
        q = validate_input(pil)
        rs = reliability_settings()
        T = float(rs["temperatures"].get("hybrid" if inc is not None else "efficientnet", 1.0))
        hc = temper(h, T)
        cconf = float(hc.max())
        withheld = []
        if safety == "Probably not a brain MRI":
            withheld.append("not a brain MRI")
        if q["verdict"] == "fail":
            withheld.append("image quality")
        if cconf < rs["threshold"]:
            withheld.append("low calibrated confidence")
        row = {"file": name, "prediction": S.class_names[hi], "confidence": float(h[hi] * 100),
               "efficientnet": S.class_names[ei], "efficientnet_conf": float(e[ei] * 100),
               "entropy": entropy_normalized(h), "colourfulness": colour, "ood_ratio": ratio,
               "safety": safety, "probabilities": _probs_dict(h),
               "quality": q["verdict"], "quality_issues": [c["name"] for c in q["checks"] if c["status"] != "ok"],
               "calibrated_confidence": cconf * 100,
               "decision": "Withheld" if withheld else "Reported", "withheld_because": withheld}
        if inc is not None:
            ii = int(np.argmax(inc[0]))
            row.update({"inception": S.class_names[ii], "inception_conf": float(inc[0][ii] * 100),
                        "agree": ii == ei})
        else:
            row.update({"inception": None, "inception_conf": None, "agree": True})
        th = pil.copy()
        th.thumbnail((96, 96))
        row["thumb"] = _png_data_url(th, "JPEG")
        rows.append(row)
    return rows


# ---------------- model card data ----------------

def model_info():
    out = []
    for key, label, model, size, scale in _eval_models():
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        feat_dim = None
        try:
            feat, _ = _get_split({"efficientnet": "eff", "inception": "inc", "densenet": "dense"}[key])
            feat_dim = int(feat.output.shape[-1]) if feat is not None else None
        except Exception:
            pass
        item = {"key": key, "name": label, "source": S.sources.get(key, "notebook variable"),
                "input": f"{size[0]} x {size[1]}", "preprocessing": "raw 0-255" if scale == 1.0 else "scaled 0-1",
                "parameters": int(model.count_params()), "trainable": trainable, "layers": len(model.layers),
                "feature_dim": feat_dim}
        if S.eval_result and key in S.eval_result.get("models", {}):
            m = S.eval_result["models"][key]
            item["evaluation"] = {k: m[k] for k in ("accuracy", "macro_f1", "macro_auc", "ece")}
            b = (S.eval_result.get("benchmark") or {}).get(key)
            if b:
                item["benchmark"] = b
            c = ((S.eval_result.get("calibration") or {}).get("models") or {}).get(key)
            if c:
                item["temperature"] = c["temperature"]
        out.append(item)
    counts = {}
    for split, d in (("training", S.train_dir), ("testing", S.test_dir)):
        if d:
            counts[split] = {c: len([f for f in os.listdir(os.path.join(d, c)) if f.lower().endswith(ALLOWED_EXT)])
                             for c in S.class_names if os.path.isdir(os.path.join(d, c))}
    return {"models": out, "dataset": counts,
            "evaluation_available": bool(S.eval_result),
            "hybrid": ({k: S.eval_result["models"]["hybrid"][k] for k in ("accuracy", "macro_f1", "macro_auc", "ece")}
                       if S.eval_result and "hybrid" in S.eval_result.get("models", {}) else None)}


# ================================================================
# TOOL 10 — TUMOUR SEGMENTATION (U-Net trained in the Tool 10 cell)
#   Data: any 2D MRI dataset with tumour masks (Kaggle images/ + masks/,
#   LGG *_mask.tif, or figshare .mat files). Inference, measurements and
#   the "do the heatmaps point at the real tumour?" evaluation live here.
# ================================================================
UNET_FILES = ["BrainTumor_UNet.keras"]
UNET_METRICS_FILE = "BrainTumor_UNet_metrics.json"
XAI_MASK_FILE = "neuronova_xai_vs_masks.json"
SEG_ROOT = "/content/seg_dataset"
SEG_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SEG_THRESHOLD = 0.5
_MASK_DIRS = {"mask", "masks", "label", "labels", "seg", "segmentation", "segmentations", "annotation",
              "annotations", "gt", "ground_truth", "groundtruth", "tumor_mask", "tumour_mask"}
_MASK_SUFFIXES = ("_mask", "-mask", " mask", "_seg", "-seg", "_label", "_gt", "_tumor", "_tumour")
_TYPE_WORDS = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary",
               "notumor": "notumor", "no_tumor": "notumor", "no tumor": "notumor"}


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def _is_mask_path(p, root):
    st = _stem(p).lower()
    if any(st.endswith(sfx) for sfx in _MASK_SUFFIXES):
        return True
    rel = os.path.relpath(os.path.dirname(p), root).lower().replace("\\", "/").split("/")
    return any(part in _MASK_DIRS for part in rel)


def _label_from_path(p):
    low = p.lower().replace("\\", "/")
    for word, lab in _TYPE_WORDS.items():
        if word in low:
            return lab
    return None


def _mat_to_png(path, out_dir):
    """figshare brain-tumour .mat (cjdata.image / tumorMask / label / PID) -> PNG image + PNG mask."""
    name = _stem(path)
    img_p, msk_p = os.path.join(out_dir, "images", name + ".png"), os.path.join(out_dir, "masks", name + ".png")
    meta_p = os.path.join(out_dir, "meta", name + ".json")
    if os.path.exists(img_p) and os.path.exists(msk_p) and os.path.exists(meta_p):
        with open(meta_p) as fh:
            return json.load(fh)
    try:
        import h5py
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            img = np.array(cj["image"]).T
            msk = np.array(cj["tumorMask"]).T
            label = int(np.array(cj["label"]).ravel()[0])
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel()) if "PID" in cj else ""
    except (OSError, ImportError, KeyError):
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)["cjdata"]
        img, msk, label, pid = m.image, m.tumorMask, int(m.label), str(getattr(m, "PID", ""))
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255
    for sub in ("images", "masks", "meta"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(img_p)
    Image.fromarray((np.asarray(msk) > 0).astype(np.uint8) * 255).save(msk_p)
    rec = {"image": img_p, "mask": msk_p, "label": {1: "meningioma", 2: "glioma", 3: "pituitary"}.get(label),
           "patient": pid}
    with open(meta_p, "w") as fh:
        json.dump(rec, fh)
    return rec


def find_segmentation_pairs(root, verbose=True):
    """Pair every MRI image with its tumour mask, whatever the folder layout."""
    images, masks, mats = [], [], []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("__MACOSX", "_converted")]
        for f in files:
            if f.startswith("._"):
                continue
            p = os.path.join(d, f)
            low = f.lower()
            if low.endswith(".mat"):
                mats.append(p)
            elif low.endswith(SEG_IMG_EXT):
                (masks if _is_mask_path(p, root) else images).append(p)
    pairs = []
    if mats:
        conv = os.path.join(root, "_converted")
        for i, m in enumerate(sorted(mats)):
            try:
                pairs.append(_mat_to_png(m, conv))
            except Exception as exc:
                if verbose:
                    print(f"  skipped {os.path.basename(m)}: {type(exc).__name__}: {exc}")
            if verbose and (i + 1) % 500 == 0:
                print(f"  converted {i + 1}/{len(mats)} .mat files")
    by_stem = {}
    for p in images:
        by_stem.setdefault(_stem(p).lower(), []).append(p)
    seen = set()
    for m in sorted(masks):
        st = _stem(m).lower()
        cands = [st] + [st[: -len(sfx)] for sfx in _MASK_SUFFIXES if st.endswith(sfx)]
        hits = [p for c in cands for p in by_stem.get(c, [])]
        if not hits:
            continue
        best = max(hits, key=lambda p: len(os.path.commonpath([os.path.abspath(p), os.path.abspath(m)])))
        if best in seen:
            continue
        seen.add(best)
        parent = os.path.basename(os.path.dirname(best))
        pairs.append({"image": best, "mask": m, "label": _label_from_path(best),
                      "patient": parent if parent.upper().startswith("TCGA") else ""})
    if verbose:
        labs = {}
        for p in pairs:
            labs[p["label"] or "unknown"] = labs.get(p["label"] or "unknown", 0) + 1
        print(f"  Found {len(pairs)} image/mask pairs ({len(mats)} .mat files, {len(masks)} mask images) "
              f"· tumour types: {labs}")
    return pairs


def prepare_segmentation_data(src, dest=SEG_ROOT, verbose=True):
    """Unzip the segmentation dataset once (from Drive) and return its image/mask pairs."""
    import zipfile
    if os.path.isdir(src):
        return find_segmentation_pairs(src, verbose)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found. Upload the segmentation dataset (zip) to your Drive folder first.")
    marker = os.path.join(dest, ".unzipped_from")
    if not (os.path.exists(marker) and open(marker).read() == f"{src}|{os.path.getsize(src)}"):
        if verbose:
            print(f"  Unzipping {os.path.basename(src)} ({os.path.getsize(src) / 1e6:.0f} MB) to {dest} ...")
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
        with open(marker, "w") as fh:
            fh.write(f"{src}|{os.path.getsize(src)}")
    elif verbose:
        print(f"  Already unzipped in {dest}")
    return find_segmentation_pairs(dest, verbose)


def split_pairs(pairs, seed=42, val=0.15, test=0.15):
    """Train / validation / test split. Slices of one patient stay together when the patient is known."""
    groups = {}
    for p in pairs:
        groups.setdefault(p.get("patient") or p["image"], []).append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(pairs)
    out = {"train": [], "val": [], "test": []}
    for k in keys:
        if len(out["test"]) < test * n:
            out["test"] += groups[k]
        elif len(out["val"]) < val * n:
            out["val"] += groups[k]
        else:
            out["train"] += groups[k]
    return out


def load_segmentation_arrays(pairs, size, verbose=True):
    """Images (greyscale) and binary masks as uint8 arrays of shape (N, size, size, 1)."""
    X = np.zeros((len(pairs), size, size, 1), dtype=np.uint8)
    Y = np.zeros((len(pairs), size, size, 1), dtype=np.uint8)
    for i, p in enumerate(pairs):
        X[i, ..., 0] = np.asarray(Image.open(p["image"]).convert("L").resize((size, size), Image.Resampling.BILINEAR))
        Y[i, ..., 0] = np.asarray(Image.open(p["mask"]).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 127
        if verbose and (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{len(pairs)}")
    return X, Y


def segmentation_scores(y_true, y_prob, thr=SEG_THRESHOLD):
    """Dice, IoU, precision, recall per image (empty-vs-empty counts as perfect) and over all pixels."""
    t = y_true.reshape(len(y_true), -1) > 0
    p = y_prob.reshape(len(y_prob), -1) >= thr
    inter = (t & p).sum(1).astype(np.float64)
    ts, ps = t.sum(1).astype(np.float64), p.sum(1).astype(np.float64)
    union = ts + ps - inter
    both_empty = (ts == 0) & (ps == 0)
    dice = np.where(both_empty, 1.0, 2 * inter / np.maximum(ts + ps, 1))
    iou = np.where(both_empty, 1.0, inter / np.maximum(union, 1))
    prec = np.where(ps > 0, inter / np.maximum(ps, 1), np.where(ts == 0, 1.0, 0.0))
    rec = np.where(ts > 0, inter / np.maximum(ts, 1), 1.0)
    has = ts > 0
    I, T_, P_ = inter.sum(), ts.sum(), ps.sum()
    return {
        "n": int(len(t)), "n_with_tumour": int(has.sum()),
        "dice": float(dice.mean()), "iou": float(iou.mean()),
        "dice_tumour_only": float(dice[has].mean()) if has.any() else None,
        "iou_tumour_only": float(iou[has].mean()) if has.any() else None,
        "precision": float(prec.mean()), "recall": float(rec.mean()),
        "global_dice": float(2 * I / max(T_ + P_, 1)), "global_iou": float(I / max(T_ + P_ - I, 1)),
        "pixel_accuracy": float(((t == p).sum()) / t.size),
        "detection_accuracy": float(((ts > 0) == (ps > 0)).mean()),
        "per_image_dice": [float(v) for v in dice],
    }


def _setup_unet():
    S.unet, S.unet_meta, S.unet_path = None, None, None
    dirs = [os.getcwd(), "/content", _HERE, DRIVE_FOLDER]
    for d in dirs:
        p = os.path.join(d, UNET_FILES[0])
        if os.path.exists(p):
            try:
                S.unet = _keras().models.load_model(p, compile=False)
                S.unet_path = p
            except Exception as exc:
                print(f"  U-Net             : could not load {p}: {exc}")
            break
    for d in dirs:
        p = os.path.join(d, UNET_METRICS_FILE)
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    S.unet_meta = json.load(fh)
            except Exception:
                pass
            break
    if S.unet is not None:
        m = S.unet_meta or {}
        dice = (m.get("test") or {}).get("dice")
        print(f"  U-Net (Tool 10)   : {S.unet_path}" + (f" · test Dice {dice:.3f}" if dice is not None else ""))
    S.xai_mask_eval = None
    p = _cache_path(XAI_MASK_FILE)
    if p and os.path.exists(p):
        try:
            with open(p) as fh:
                S.xai_mask_eval = json.load(fh)
        except Exception:
            pass


def segment_probability(pil):
    """U-Net tumour probability map at the U-Net's input size."""
    size = int(S.unet.input_shape[1])
    ch = int(S.unet.input_shape[-1])
    g = np.asarray(pil.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    x = np.repeat(g[None, ..., None], ch, axis=-1)
    return np.asarray(S.unet(x, training=False))[0, ..., 0]


def _otsu(a):
    hist, edges = np.histogram(a, bins=256, range=(0, 256))
    w = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    total, mt = w[-1], mu[-1]
    w2 = total - w
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w - mu * total) ** 2 / (w * w2)
    between[~np.isfinite(between)] = 0
    return int(np.argmax(between))


def brain_mask(pil):
    """Rough brain/head area: Otsu threshold, holes filled, largest connected region."""
    from scipy import ndimage
    g = np.asarray(pil.convert("L"), dtype=np.float32)
    thr = max(10, min(_otsu(g), 80)) * 0.5
    m = ndimage.binary_fill_holes(g > thr)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m


def _mask_overlay(base_rgb, mask, prob=None, colour=(95, 225, 200)):
    from scipy import ndimage
    out = base_rgb.astype(np.float32).copy()
    if mask.any():
        c = np.array(colour, dtype=np.float32)
        out[mask] = out[mask] * 0.55 + c * 0.45
        edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
        out[edge] = c
    return np.clip(out, 0, 255).astype(np.uint8)


def stage_segment(case):
    t0 = time.perf_counter()
    if S.unet is None:
        return {"available": False,
                "message": "No U-Net model yet. Run the Tool 10 cell once to train it on a dataset with tumour masks.",
                "ms": _ms(t0)}
    from scipy import ndimage
    img = case["image"].convert("RGB")
    W, H = img.size
    with S.lock:
        prob = segment_probability(img)
    prob_full = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR),
                           dtype=np.float32) / 255.0
    raw = prob_full >= SEG_THRESHOLD
    lab, n = ndimage.label(raw)
    min_px = max(20, int(0.0008 * W * H))
    keep = []
    if n:
        sizes = ndimage.sum(raw, lab, range(1, n + 1))
        keep = [i + 1 for i, s_ in enumerate(sizes) if s_ >= min_px]
    mask = ndimage.binary_fill_holes(np.isin(lab, keep)) if keep else np.zeros_like(raw)
    area = int(mask.sum())
    detected = area > 0
    bm = brain_mask(img)
    brain_area = int(bm.sum())
    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if detected else None
    centroid = [float(xs.mean()), float(ys.mean())] if detected else None
    side = "left side of the image" if detected and centroid[0] < W / 2 else "right side of the image"
    vert = "upper" if detected and centroid[1] < H / 3 else ("lower" if detected and centroid[1] > 2 * H / 3 else "middle")
    pred = None
    if "hybrid_probs" in case:
        pred = S.class_names[int(np.argmax(case["hybrid_probs"]))]
    elif "classify" in case["results"]:
        pred = case["results"]["classify"]["prediction"]
    tumour_class = pred is not None and pred.lower() not in ("notumor", "no_tumor", "no tumour")
    consistent = None if pred is None else (detected == tumour_class)
    att = None
    hm = case.get("heatmap_resized")
    if detected and hm is not None and hm.shape == mask.shape and hm.max() > 0:
        py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
        dil = ndimage.binary_dilation(mask, iterations=max(2, int(0.02 * max(W, H))))
        k = max(1, area)
        thr = np.partition(hm.ravel(), -k)[-k]
        top = hm >= thr
        att = {"peak_inside": bool(dil[py, px]), "energy_inside": float(hm[mask].sum() / max(hm.sum(), 1e-9)),
               "iou_top_region": float((top & mask).sum() / max((top | mask).sum(), 1)),
               "area_fraction": float(area / (W * H))}
    spacing = case.get("pixel_spacing_mm")
    dw, dh = case["display_size"]
    base = np.asarray(img.resize((dw, dh), Image.Resampling.BILINEAR), dtype=np.uint8)
    m_disp = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((dw, dh), Image.Resampling.NEAREST)) > 127
    p_disp = np.asarray(Image.fromarray((prob_full * 255).astype(np.uint8)).resize((dw, dh), Image.Resampling.BILINEAR),
                        dtype=np.float32) / 255.0
    meta = S.unet_meta or {}
    if not detected:
        text = "The U-Net found no tumour region"
    else:
        text = (f"The U-Net marked a region of {area:,} pixels ({area / (W * H) * 100:.2f}% of the image"
                + (f", {area / max(brain_area, 1) * 100:.2f}% of the brain area" if brain_area else "")
                + f") in the {vert} {side}")
    if consistent is False:
        text += (". This disagrees with the classifier, which predicted "
                 + (f"'{pred}'" if pred else "no class") + "; review both carefully")
    return {
        "available": True,
        "detected": detected,
        "area_px": area,
        "area_percent_image": float(area / (W * H) * 100),
        "brain_area_px": brain_area,
        "area_percent_brain": float(area / brain_area * 100) if brain_area else None,
        "area_mm2": float(area * spacing[0] * spacing[1]) if (spacing and detected) else None,
        "equivalent_diameter_px": float(2 * np.sqrt(area / np.pi)) if detected else None,
        "bbox": bbox,
        "centroid": centroid,
        "location": f"{vert} {side}" if detected else None,
        "components": int(len(keep)),
        "mean_probability": float(prob_full[mask].mean()) if detected else None,
        "max_probability": float(prob_full.max()),
        "classifier_prediction": pred,
        "consistent_with_classifier": consistent,
        "gradcam_vs_mask": att,
        "threshold": SEG_THRESHOLD,
        "model": {"arch": meta.get("arch", "U-Net"), "input": int(S.unet.input_shape[1]),
                  "test_dice": (meta.get("test") or {}).get("dice"), "test_iou": (meta.get("test") or {}).get("iou"),
                  "dataset": meta.get("dataset")},
        "overlay": _png_data_url(Image.fromarray(_mask_overlay(base, m_disp)), "JPEG"),
        "probability_map": _png_data_url(_overlay(base, p_disp, 0.5), "JPEG"),
        "mask": _png_data_url(Image.fromarray(m_disp.astype(np.uint8) * 255).convert("L")),
        "summary": text + ".",
        "original_size": [W, H],
        "ms": _ms(t0),
    }


def evaluate_xai_with_masks(pairs, n=80, seed=0, verbose=True, save=True):
    """Do the explanation heatmaps point at the tumour that an expert outlined?
    Pointing game (peak inside the mask), energy inside the mask, and IoU of the top region (same size as the mask),
    each compared with what a random heatmap would score."""
    from scipy import ndimage
    rng = random.Random(seed)
    cand = []
    for p in pairs:
        m = np.asarray(Image.open(p["mask"]).convert("L").resize((224, 224), Image.Resampling.NEAREST)) > 127
        if m.sum() >= 20:
            cand.append((p, m))
    rng.shuffle(cand)
    cand = cand[:n]
    if not cand:
        raise RuntimeError("No masks with a tumour were found.")
    methods = ["gradcam", "gradcam_pp", "integrated_gradients", "consensus"]
    rows = {k: {"hit": [], "energy": [], "iou": []} for k in methods}
    base = {"hit": [], "energy": [], "iou": []}
    labels, t0 = [], time.time()
    _ensure_grad_model()
    for i, (p, m) in enumerate(cand):
        img = Image.open(p["image"]).convert("RGB")
        x = np.expand_dims(_prep_eff(img), 0).astype(np.float32)
        with S.lock:
            probs = np.asarray(S.effnet(x, training=False))[0]
            idx = int(np.argmax(probs))
            heat, mx = _gradcam_raw(x, idx)
            if mx <= 0:
                heat, mx = _gradcam_raw(x, idx, use_logit=True)
            gc = _resize_map(_norm_map(heat if heat is not None else np.zeros((7, 7))))
            gpp = gradcam_plus_plus(x, idx)
            ig = integrated_gradients(x, idx)
        maps = {"gradcam": gc, "gradcam_pp": gpp, "integrated_gradients": ig}
        maps["consensus"] = _norm_map(np.mean([maps[k] for k in ("gradcam", "gradcam_pp", "integrated_gradients")], 0))
        dil = ndimage.binary_dilation(m, iterations=4)
        k = int(m.sum())
        for name, mp in maps.items():
            py, px = np.unravel_index(int(np.argmax(mp)), mp.shape)
            thr = np.partition(mp.ravel(), -k)[-k]
            top = mp >= thr
            rows[name]["hit"].append(float(dil[py, px]) if mp.max() > 0 else 0.0)
            rows[name]["energy"].append(float(mp[m].sum() / max(mp.sum(), 1e-9)))
            rows[name]["iou"].append(float((top & m).sum() / max((top | m).sum(), 1)))
        frac = k / m.size
        base["hit"].append(float(dil.mean()))
        base["energy"].append(float(frac))
        base["iou"].append(float(frac / (2 - frac)))     # expected IoU of a random region of the same size
        labels.append(p.get("label") or "unknown")
        if verbose and (i + 1) % 10 == 0:
            print(f"  explained {i + 1}/{len(cand)} MRIs ({time.time() - t0:.0f} s)", end="\r", flush=True)
    names = {"gradcam": "Grad-CAM", "gradcam_pp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients",
             "consensus": "Consensus of the three", "random": "Random heatmap (baseline)"}
    summary = {}
    for k in methods:
        summary[k] = {"label": names[k], **{f"{m_}": float(np.mean(v)) for m_, v in rows[k].items()},
                      **{f"{m_}_std": float(np.std(v)) for m_, v in rows[k].items()}}
        by = {}
        for lab in sorted(set(labels)):
            idxs = [j for j, l_ in enumerate(labels) if l_ == lab]
            by[lab] = {"n": len(idxs), "hit": float(np.mean([rows[k]["hit"][j] for j in idxs])),
                       "energy": float(np.mean([rows[k]["energy"][j] for j in idxs]))}
        summary[k]["by_type"] = by
    summary["random"] = {"label": names["random"], **{m_: float(np.mean(v)) for m_, v in base.items()}}
    res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n": len(cand), "seconds": round(time.time() - t0, 1), "methods": methods + ["random"],
           "summary": summary, "classifier": "EfficientNetV2-B0",
           "note": ("Expert masks come from the segmentation dataset. If those MRIs were also in the classifier's "
                    "training data, the classifier has seen them, but it never saw the masks.")}
    S.xai_mask_eval = res
    if save:
        p = _cache_path(XAI_MASK_FILE)
        if p:
            with open(p, "w") as fh:
                json.dump(res, fh)
    if verbose:
        print(f"  Explained {len(cand)} MRIs with expert masks in {res['seconds']} s".ljust(60))
    return res


def show_segmentation(case):
    import matplotlib.pyplot as plt
    sg = case["results"].get("segment")
    if not sg or not sg.get("available"):
        print("TOOL 10 —", (sg or {}).get("message", "segmentation has not run"))
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    _show_data_url(axes[0], case["image_payload"]["src"], "MRI")
    _show_data_url(axes[1], sg["probability_map"], "U-Net tumour probability")
    _show_data_url(axes[2], sg["overlay"], "Segmented region (threshold 0.5)")
    plt.suptitle(f"TOOL 10 — {sg['model']['arch']} segmentation")
    plt.tight_layout()
    plt.show()
    print(sg["summary"])
    if sg["detected"]:
        print(f"  Area: {sg['area_px']:,} px · {sg['area_percent_image']:.2f}% of the image"
              + (f" · {sg['area_percent_brain']:.2f}% of the brain area" if sg['area_percent_brain'] else ""))
        print(f"  Box (x0, y0, x1, y1): {sg['bbox']} · equivalent diameter {sg['equivalent_diameter_px']:.0f} px"
              f" · mean probability {sg['mean_probability']:.2f}")
        g = sg.get("gradcam_vs_mask")
        if g:
            print(f"  Grad-CAM peak inside the region: {'yes' if g['peak_inside'] else 'no'} · "
                  f"{g['energy_inside'] * 100:.0f}% of Grad-CAM energy inside · IoU of the top region {g['iou_top_region']:.2f}")


def show_xai_mask_eval(res=None):
    import matplotlib.pyplot as plt
    res = res or S.xai_mask_eval
    sm = res["summary"]
    print(f"TOOL 10B — Explanation methods vs expert tumour masks ({res['n']} MRIs, {res['classifier']})")
    print(f"  {'Method':<28}{'Pointing game':>15}{'Energy inside':>15}{'IoU top region':>16}")
    for k in res["methods"]:
        v = sm[k]
        print(f"  {v['label']:<28}{v['hit'] * 100:>14.1f}%{v['energy'] * 100:>14.1f}%{v['iou']:>16.3f}")
    print("  " + res["note"])
    ks = res["methods"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.25
    for j, (key, lab) in enumerate((("hit", "Pointing game"), ("energy", "Energy inside mask"), ("iou", "IoU top region"))):
        ax.bar(np.arange(len(ks)) + (j - 1) * w, [sm[k][key] for k in ks], w, label=lab)
    ax.set_xticks(range(len(ks)), [sm[k]["label"] for k in ks], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Do the heatmaps point at the tumour? (higher is better; last group is chance)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ================================================================
# TOOL 11 — 3D MRI VOLUMES (NIfTI .nii / .nii.gz or a DICOM series)
#   Slice-by-slice U-Net segmentation, 3D tumour volume in cm³ from the
#   voxel spacing in the file header, per-slice tumour probability from
#   the classifiers, and surface meshes for a rotatable 3D view.
#   A single 2D image cannot give a real volume, so this needs a 3D scan.
# ================================================================
VOLUME_EXT = (".nii", ".nii.gz", ".zip", ".dcm")
VOLUME_KEEP = 3
VOLUME_MESH_GRID = 96


def _load_nifti(path):
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    zx, zy, zz = (float(v) for v in img.header.get_zooms()[:3])
    axial = np.stack([np.rot90(data[:, :, k]) for k in range(data.shape[2])])    # (Z, H, W), anterior at the top
    return axial, (zz, zy, zx), {"format": "NIfTI", "voxel_mm": [zx, zy, zz], "shape": list(data.shape)}


def _load_dicom(paths):
    import pydicom
    items = []
    for p in paths:
        try:
            d = pydicom.dcmread(p, force=True)
            arr = d.pixel_array
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        items.append(d)
    if not items:
        raise ValueError("No readable DICOM images were found.")
    series = {}
    for d in items:
        series.setdefault(str(getattr(d, "SeriesInstanceUID", "single")), []).append(d)
    ds = max(series.values(), key=len)

    def zpos(d):
        try:
            return float(d.ImagePositionPatient[2])
        except Exception:
            return float(getattr(d, "InstanceNumber", 0) or 0)
    ds.sort(key=zpos)
    shape = ds[0].pixel_array.shape
    ds = [d for d in ds if d.pixel_array.shape == shape]
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1) or 1)
                    + float(getattr(d, "RescaleIntercept", 0) or 0) for d in ds])
    ps = [float(v) for v in getattr(ds[0], "PixelSpacing", [1.0, 1.0])]
    zs = [zpos(d) for d in ds]
    dz = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else 0.0
    if not dz:
        dz = float(getattr(ds[0], "SliceThickness", 1.0) or 1.0)
    info = {"format": "DICOM", "slices": len(ds), "voxel_mm": [ps[1], ps[0], dz],
            "modality": str(getattr(ds[0], "Modality", "")), "series": str(getattr(ds[0], "SeriesDescription", ""))}
    return vol, (dz, ps[0], ps[1]), info


def read_dicom_image(raw):
    """One DICOM file -> PIL image (windowed to 8 bit) + pixel spacing + safe metadata (no patient identifiers)."""
    import pydicom
    d = pydicom.dcmread(io.BytesIO(raw), force=True)
    a = d.pixel_array.astype(np.float32)
    if a.ndim == 3:
        a = a[a.shape[0] // 2] if a.shape[-1] not in (3, 4) else a.mean(-1)
    a = a * float(getattr(d, "RescaleSlope", 1) or 1) + float(getattr(d, "RescaleIntercept", 0) or 0)
    lo, hi = np.percentile(a, [0.5, 99.5])
    g = np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        g = 255 - g
    ps = [float(v) for v in getattr(d, "PixelSpacing", [])] or None
    meta = {k: str(getattr(d, k)) for k in ("Modality", "SeriesDescription", "BodyPartExamined", "SliceThickness",
                                              "MagneticFieldStrength", "Manufacturer") if hasattr(d, k)}
    return Image.fromarray(g.astype(np.uint8)).convert("RGB"), ps, meta


def _norm_uint8(vol):
    nz = vol[vol > np.percentile(vol, 5)] if (vol > 0).mean() < 0.99 else vol
    lo, hi = np.percentile(nz if nz.size else vol, [0.5, 99.5])
    return (np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255).astype(np.uint8)


def _brain_mask_3d(V):
    from scipy import ndimage
    thr = max(10, min(_otsu(V[V > 5]) if (V > 5).any() else 40, 80)) * 0.5
    m = V > thr
    m = np.stack([ndimage.binary_fill_holes(s) for s in m])
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m


def _mesh(mask, spacing, grid=VOLUME_MESH_GRID, step=1, sigma=1.0, crop=False):
    from scipy import ndimage
    from skimage import measure
    if not mask.any():
        return None
    off = np.zeros(3)
    if crop:
        zz, hh, ww = np.where(mask)
        lo = np.maximum([zz.min() - 3, hh.min() - 3, ww.min() - 3], 0)
        hi = np.minimum([zz.max() + 4, hh.max() + 4, ww.max() + 4], mask.shape)
        mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        off = lo.astype(np.float64)
    f = max(1, int(np.ceil(max(mask.shape) / grid)))
    m = ndimage.gaussian_filter(mask.astype(np.float32), sigma)[::f, ::f, ::f]
    m = np.pad(m, 1)
    if m.max() < 0.5:
        return None
    verts, faces, _, _ = measure.marching_cubes(m, 0.5, step_size=step)
    verts = (verts - 1) * f + off
    verts = verts * np.asarray(spacing, dtype=np.float64)
    return verts.astype(np.float32), faces.astype(np.uint32)


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def process_volume(path=None, paths=None, name="volume", seg_path=None, verbose=False):
    """Load a 3D MRI, segment the tumour slice by slice and measure it in 3D."""
    from scipy import ndimage
    t0 = time.perf_counter()
    if paths:
        A, spacing, info = _load_dicom(paths)
    elif path and path.lower().endswith(".zip"):
        import zipfile, tempfile
        tmp = tempfile.mkdtemp(prefix="neuronova_dcm_")
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        files = [os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs if not f.startswith(".")]
        nii = [f for f in files if f.lower().endswith((".nii", ".nii.gz"))]
        if nii:
            A, spacing, info = _load_nifti(nii[0])
        else:
            A, spacing, info = _load_dicom(files)
        shutil.rmtree(tmp, ignore_errors=True)
    elif path and path.lower().endswith(".dcm"):
        A, spacing, info = _load_dicom([path])
    else:
        A, spacing, info = _load_nifti(path)
    if A.shape[0] < 3:
        raise ValueError("This file has fewer than 3 slices, so it is not a 3D volume. Use the normal upload for 2D images.")
    V = _norm_uint8(A)
    Z, H, W = V.shape
    if verbose:
        print(f"  Volume {Z} slices of {W} x {H} px, voxel {spacing[2]:.2f} x {spacing[1]:.2f} x {spacing[0]:.2f} mm")
    head = _brain_mask_3d(V)
    er = max(1, int(round(4.0 / max(float(spacing[1]), 0.1))))       # drop about 4 mm of skull and scalp
    brain = np.stack([ndimage.binary_erosion(sl, iterations=er) for sl in head])
    barea = brain.reshape(Z, -1).sum(1)
    # skip the top and bottom slices that only cut through the skull cap or the skull base
    use = np.where(barea > max(0.02 * H * W, 0.2 * barea.max()))[0]
    tumour = np.zeros_like(brain)
    prob_max = np.zeros(Z, dtype=np.float32)
    if S.unet is not None and len(use):
        size = int(S.unet.input_shape[1])
        ch = int(S.unet.input_shape[-1])
        for s0 in range(0, len(use), 32):
            ks = use[s0:s0 + 32]
            x = np.stack([np.asarray(Image.fromarray(V[k]).resize((size, size), Image.Resampling.BILINEAR),
                                     dtype=np.float32) / 255.0 for k in ks])[..., None]
            with S.lock:
                pr = np.asarray(S.unet(np.repeat(x, ch, -1), training=False))[..., 0]
            for j, k in enumerate(ks):
                p = np.asarray(Image.fromarray((pr[j] * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR),
                               dtype=np.float32) / 255.0
                tumour[k] = p >= SEG_THRESHOLD
                prob_max[k] = float(p.max())
            if verbose:
                print(f"  segmented {min(len(use), s0 + 32)}/{len(use)} slices", end="\r", flush=True)
        tumour &= brain
        lab, n = ndimage.label(tumour)
        if n:
            # drop thin shells that hug the skull (bright skull or fat cut at an angle), keep compact lesions
            dist = ndimage.distance_transform_edt(head, sampling=spacing)
            good = []
            for i, sl in enumerate(ndimage.find_objects(lab)):
                comp = lab[sl] == (i + 1)
                fill = comp.sum() / comp.size
                near = float(np.median(dist[sl][comp]))
                if not (near < 7.0 and fill < 0.25):
                    good.append(i + 1)
            tumour = np.isin(lab, good)
            lab, n = ndimage.label(tumour)
        if n:
            sizes = ndimage.sum(tumour, lab, range(1, n + 1))
            min_vox = max(30, int(0.0005 * brain.sum()), int(0.25 * sizes.max()))   # main lesion + large satellites
            keep = [i + 1 for i, v in enumerate(sizes) if v >= min_vox]
            tumour = np.isin(lab, keep)
    # per-slice classifier opinion (hybrid vote on each slice with brain)
    cls_prob = np.full(Z, np.nan, dtype=np.float32)
    cls_pred = [None] * Z
    if len(use):
        nt = [i for i, c in enumerate(S.class_names) if c.lower() in ("notumor", "no_tumor", "no tumour")]
        for s0 in range(0, len(use), 16):
            ks = use[s0:s0 + 16]
            pils = [Image.fromarray(V[k]).convert("RGB") for k in ks]
            with S.lock:
                hyb, _, _ = predict_hybrid(pils)
            for j, k in enumerate(ks):
                cls_prob[k] = float(1 - hyb[j][nt[0]]) if nt else float(hyb[j].max())
                cls_pred[k] = S.class_names[int(np.argmax(hyb[j]))]
    vox = float(spacing[0] * spacing[1] * spacing[2])
    tvox = int(tumour.sum())
    t_area = tumour.reshape(Z, -1).sum(1)
    stats = {"tumour_voxels": tvox, "tumour_cm3": tvox * vox / 1000.0, "brain_cm3": float(brain.sum() * vox / 1000.0),
             "tumour_percent_brain": float(tvox / max(brain.sum(), 1) * 100), "slices": int(Z),
             "slices_with_tumour": int((t_area > 0).sum()), "voxel_mm3": vox}
    if tvox:
        zz, hh, ww = np.where(tumour)
        ext = [(zz.max() - zz.min() + 1) * spacing[0], (hh.max() - hh.min() + 1) * spacing[1],
               (ww.max() - ww.min() + 1) * spacing[2]]
        stats.update({"extent_mm": {"superior_inferior": float(ext[0]), "anterior_posterior": float(ext[1]),
                                    "left_right": float(ext[2])},
                      "largest_extent_mm": float(max(ext)),
                      "centroid_slice": int(round(zz.mean())), "peak_slice": int(np.argmax(t_area)),
                      "components": int(ndimage.label(tumour)[1])})
        best = int(np.argmax(t_area))
        stats["type_from_peak_slice"] = cls_pred[best]
    gt_scores = None
    G = None
    if seg_path:
        try:
            Ga, _, _ = _load_nifti(seg_path)
            G = Ga > 0
            if G.shape == tumour.shape:
                inter = float((G & tumour).sum())
                gs, ts = float(G.sum()), float(tumour.sum())
                gt_scores = {"dice": 2 * inter / max(gs + ts, 1), "iou": inter / max(gs + ts - inter, 1),
                             "gt_cm3": gs * vox / 1000.0}
            else:
                gt_scores = {"error": f"Mask shape {list(G.shape)} does not match the volume {list(tumour.shape)}."}
                G = None
        except Exception as exc:
            gt_scores = {"error": f"{type(exc).__name__}: {exc}"}
    centre = np.array([Z / 2 * spacing[0], H / 2 * spacing[1], W / 2 * spacing[2]])
    meshes = {}
    for key, m_, kw in (("brain", brain, {"step": 2, "sigma": 1.5}), ("tumour", tumour, {"step": 1, "sigma": 0.8, "crop": True}),
                        ("ground_truth", G, {"step": 1, "sigma": 0.8, "crop": True})):
        if m_ is None or not m_.any():
            continue
        res = _mesh(m_, spacing, **kw)
        if res is None:
            continue
        v, f = res
        v = v - centre
        meshes[key] = {"vertices": _b64((v[:, [2, 0, 1]] * np.array([1, 1, -1])).astype(np.float32)),
                       "faces": _b64(f.astype(np.uint32)), "n_vertices": int(len(v)), "n_faces": int(len(f))}
    vid = "VOL-" + uuid.uuid4().hex[:8].upper()
    S.volumes[vid] = {"V": V, "tumour": tumour, "gt": G, "spacing": spacing, "name": name}
    while len(S.volumes) > VOLUME_KEEP:
        S.volumes.pop(next(iter(S.volumes)))
    per_slice = [{"i": int(k), "tumour_px": int(t_area[k]), "brain_px": int(barea[k]),
                  "tumour_prob": None if np.isnan(cls_prob[k]) else float(cls_prob[k]), "class": cls_pred[k],
                  "unet_max": float(prob_max[k])} for k in range(Z)]
    return {"volume_id": vid, "name": name, "info": info, "shape": [int(Z), int(H), int(W)],
            "spacing_mm": [float(x) for x in spacing], "stats": stats, "ground_truth": gt_scores,
            "segmentation": "U-Net (Tool 10), slice by slice" if S.unet is not None else None,
            "message": None if S.unet is not None else "No U-Net yet (Tool 10), so only the brain surface is shown.",
            "meshes": meshes, "per_slice": per_slice, "ms": _ms(t0),
            "start_slice": int(stats.get("peak_slice", Z // 2))}


def volume_slice(vid, plane="axial", i=None, overlay=True, size=384):
    v = S.volumes.get(vid)
    if v is None:
        raise KeyError("Volume not found. Upload it again.")
    V, T, G, sp = v["V"], v["tumour"], v["gt"], v["spacing"]
    Z, H, W = V.shape
    if plane == "coronal":
        n = H
        i = n // 2 if i is None else int(np.clip(i, 0, n - 1))
        g, t = np.flipud(V[:, i, :]), np.flipud(T[:, i, :])
        gt = np.flipud(G[:, i, :]) if G is not None else None
        aspect = sp[0] / sp[2]
    elif plane == "sagittal":
        n = W
        i = n // 2 if i is None else int(np.clip(i, 0, n - 1))
        g, t = np.flipud(V[:, :, i]), np.flipud(T[:, :, i])
        gt = np.flipud(G[:, :, i]) if G is not None else None
        aspect = sp[0] / sp[1]
    else:
        n = Z
        i = n // 2 if i is None else int(np.clip(i, 0, n - 1))
        g, t = V[i], T[i]
        gt = G[i] if G is not None else None
        aspect = sp[1] / sp[2]
    rgb = np.repeat(g[..., None], 3, -1)
    if overlay:
        if gt is not None and gt.any():
            from scipy import ndimage
            e = gt & ~ndimage.binary_erosion(gt, iterations=1)
            rgb = rgb.copy()
            rgb[e] = (255, 193, 112)
        rgb = _mask_overlay(rgb, t)
    im = Image.fromarray(rgb.astype(np.uint8))
    h_px = max(1, int(round(im.size[1] * aspect)))
    im = im.resize((im.size[0], h_px), Image.Resampling.BILINEAR)
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue(), n, i


def show_volume(res):
    """Three planes through the tumour, the per-slice profile and an interactive 3D view (plotly)."""
    import matplotlib.pyplot as plt
    st = res["stats"]
    print(f"TOOL 11 — {res['name']} ({res['info']['format']}, {res['shape'][0]} slices, "
          f"voxel {res['spacing_mm'][2]:.2f} x {res['spacing_mm'][1]:.2f} x {res['spacing_mm'][0]:.2f} mm)")
    if res.get("message"):
        print("  " + res["message"])
    print(f"  Brain volume  : {st['brain_cm3']:.1f} cm3")
    print(f"  Tumour volume : {st['tumour_cm3']:.2f} cm3 ({st['tumour_percent_brain']:.2f}% of the brain) "
          f"in {st['slices_with_tumour']} of {st['slices']} slices")
    if st.get("extent_mm"):
        e = st["extent_mm"]
        print(f"  Extent        : {e['left_right']:.0f} (L-R) x {e['anterior_posterior']:.0f} (A-P) x "
              f"{e['superior_inferior']:.0f} (S-I) mm; largest {st['largest_extent_mm']:.0f} mm")
        print(f"  Classifier on the slice with the most tumour: {st.get('type_from_peak_slice')}")
    if res.get("ground_truth") and "dice" in res["ground_truth"]:
        g = res["ground_truth"]
        print(f"  Against the expert mask: 3D Dice {g['dice']:.3f}, IoU {g['iou']:.3f} (expert volume {g['gt_cm3']:.2f} cm3)")
    vid, k = res["volume_id"], res["start_slice"]
    v = S.volumes[vid]
    zz = np.where(v["tumour"].any(axis=(0, 1)))[0]
    hh = np.where(v["tumour"].any(axis=(0, 2)))[0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    for ax, (plane, idx) in zip(axes[:3], (("axial", k), ("coronal", int(hh.mean()) if len(hh) else None),
                                           ("sagittal", int(zz.mean()) if len(zz) else None))):
        raw, n, i = volume_slice(vid, plane, idx)
        ax.imshow(Image.open(io.BytesIO(raw)))
        ax.set_title(f"{plane} slice {i + 1}/{n}", fontsize=10)
        ax.axis("off")
    ps = res["per_slice"]
    axes[3].plot([p["tumour_px"] for p in ps], color="#21A3BC", label="U-Net tumour pixels")
    ax2 = axes[3].twinx()
    ax2.plot([p["tumour_prob"] if p["tumour_prob"] is not None else np.nan for p in ps], color="#B63325", lw=1,
             label="classifier: P(tumour)")
    ax2.set_ylim(0, 1)
    axes[3].set_xlabel("Axial slice")
    axes[3].set_title("Tumour across slices")
    axes[3].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
    try:
        import plotly.graph_objects as go
        data = []
        for key, colour, op in (("brain", "lightgrey", 0.12), ("tumour", "#5FE1C8", 1.0), ("ground_truth", "#FFC170", 0.35)):
            m = res["meshes"].get(key)
            if not m:
                continue
            vtx = np.frombuffer(base64.b64decode(m["vertices"]), dtype=np.float32).reshape(-1, 3)
            fc = np.frombuffer(base64.b64decode(m["faces"]), dtype=np.uint32).reshape(-1, 3)
            data.append(go.Mesh3d(x=vtx[:, 0], y=vtx[:, 1], z=vtx[:, 2], i=fc[:, 0], j=fc[:, 1], k=fc[:, 2],
                                  color=colour, opacity=op, name=key.replace("_", " ")))
        fig3 = go.Figure(data=data)
        fig3.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                                      camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.5, y=0.5, z=1.5))),
                           title="TOOL 11 — 3D view (drag to rotate, scroll to zoom)", height=560,
                           margin=dict(l=0, r=0, t=40, b=0))
        fig3.show()
    except ImportError:
        print("  (install plotly for the interactive 3D view)")


# ================================================================
# TOOL 13 — MEDICAL RAG WITH SOURCE CITATIONS
#   Answers come from a controlled set of documents (your PDFs / text in
#   brain_tumor_models/medical_docs/ plus a built-in NeuroNova handbook):
#   page-level chunks -> BM25 keyword search + embedding search, fused
#   by reciprocal rank -> Gemini writes an answer that may only use the
#   numbered passages and must cite them [n]. Every cited sentence is
#   checked against its passage, and the answer says so when the
#   sources do not cover the question.
# ================================================================
RAG_DIR_NAME = "medical_docs"
RAG_EXT = (".pdf", ".txt", ".md")
RAG_EMBED_MODELS = ("all-MiniLM-L6-v2", "gemini-embedding-001")
_STOP = set("""a an the and or of to in on for with without by from at as is are was were be been being this that these
those it its into than then there their them they he she his her you your we our i me my can could may might will
would should shall do does did done not no yes if but so such also more most less least very about over under after
before between within what which who whom whose when where why how all any each other some many much one two three
per via vs come comes go goes get gets make makes mean means tell know show give take find happen need want help
explain describe work works use used using like please""".split())

HANDBOOK = [
    ("Overview of NeuroNova", """NeuroNova is a student research project for brain tumour MRI analysis. It is not a medical device and its output is not a diagnosis.
A 2D MRI slice is classified into four classes: glioma, meningioma, pituitary tumour and no tumour. EfficientNetV2-B0 (224 x 224 input) is the main classifier (Tool 1). InceptionV3 (299 x 299 input) gives a second opinion, and the two softmax outputs are averaged with equal weights in a hybrid soft vote (Tool 3). An optional DenseNet121 trained with a separate validation split forms a three-model ensemble (Tool 9).
Every prediction comes with evidence: Grad-CAM heatmaps (Tool 2), a stability test under small image changes (Tool 3B), occlusion sensitivity compared with Grad-CAM (Tool 3C), a reliability grade (Tool 3D), uncertainty and input validation (Tool 8A), an explainability lab (Tool 8B), similar training scans (Tool 8C), U-Net tumour segmentation (Tool 10) and 3D volume measurement (Tool 11)."""),
    ("Explainability methods", """Grad-CAM weights the feature maps of the last convolutional layer by the average gradient of the predicted class and keeps the positive part. It is a coarse map (7 x 7 for EfficientNetV2-B0) scaled up to the image, so it marks model attention, not an exact tumour boundary.
Grad-CAM++ weights each spatial location separately with higher-order gradient terms, which separates several regions better. Integrated Gradients adds up input gradients along a straight path from a black image to the scan; it satisfies completeness, meaning the attributions sum to the change in the class score. LIME hides random groups of superpixels and fits a weighted linear model to see which regions matter.
Occlusion sensitivity slides a grey patch over the image and measures how much the hybrid confidence drops; it needs no gradients, so agreement with Grad-CAM (IoU and correlation of the top 20% regions) is independent evidence.
Explanations can be checked against expert tumour masks (Tool 10B): the pointing game asks whether the hottest pixel lies inside the tumour, and the energy score measures the share of the heatmap inside the tumour, both compared with a random heatmap. A deletion test removes the most important pixels first; a faithful explanation makes the confidence fall quickly."""),
    ("Uncertainty, input validation and the reject option", """Monte Carlo dropout keeps dropout active at prediction time and runs the classifier head 50 times. The mean is the prediction; the spread splits into epistemic (model) uncertainty, which shrinks with more training data, and aleatoric (data) uncertainty, which comes from noise or ambiguity in the image.
Input validation runs before a prediction is trusted. It checks image size, shape, colour (MRI slices are greyscale), sharpness (variance of the Laplacian), contrast and brightness against the training scans, and measures how far the CNN features are from the training scans (Mahalanobis distance in a whitened PCA space, compared with the 99th percentile of held-out training distances). Images far outside are flagged as probably not a brain MRI.
Temperature scaling divides the logits by a temperature T fitted on a calibration set to minimise the negative log-likelihood. T above 1 softens over-confident probabilities and T below 1 sharpens under-confident ones; the expected calibration error (ECE) measures the gap between confidence and accuracy.
The reject option withholds a class when the calibrated confidence is below a threshold, when the image fails the quality checks, or when it does not look like a brain MRI. The threshold is chosen on one half of the Testing folder so that the reported cases have half the error rate, and it is scored on the other half (coverage and selective accuracy)."""),
    ("Segmentation and 3D measurement", """Tool 10 trains a U-Net or Attention U-Net on MRI slices that come with expert tumour masks. The loss is binary cross-entropy plus Dice loss. The Dice coefficient is 2|A and B| / (|A| + |B|) and IoU is |A and B| / |A or B|; both are 1 for a perfect outline. Results are reported on a held-out test split, split by patient when the patient is known.
Attention gates let the decoder weight skip-connection features by a learned spatial attention map, which suppresses irrelevant regions.
Tool 11 applies the 2D U-Net to every slice of a 3D scan (NIfTI or a DICOM series), removes thin shells that hug the skull and tiny fragments, and measures the tumour volume as voxel count times voxel volume from the file header. Surface meshes are built with the marching cubes algorithm. Measurements from a 2D image are in pixels; real sizes need the scanner's pixel spacing."""),
    ("Evaluation metrics", """Accuracy is the share of correct predictions. Precision for a class is the share of predictions of that class that are correct; recall (sensitivity) is the share of true cases of that class that the model finds; F1 is their harmonic mean. Macro averages give every class the same weight.
ROC-AUC measures how well the class probability ranks positives above negatives, one class against the rest; 0.5 is chance and 1 is perfect. Average precision summarises the precision-recall curve and is more informative than ROC-AUC when positives are rare.
Cohen's kappa measures agreement beyond chance and the Matthews correlation coefficient (MCC) is a balanced correlation between true and predicted classes from -1 to 1. The Brier score and log loss measure the quality of the probabilities themselves.
In this project the Testing folder was also used for early stopping of EfficientNetV2-B0 and InceptionV3, so their test scores are likely a little optimistic. A clean protocol keeps a separate validation split for model selection and touches the test set once, as Tool 9 does."""),
    ("Limitations and safe use", """The models know only four classes and always output one of them, even for unrelated images; input validation and the reject option reduce, but do not remove, this risk. Performance depends on how similar new scans are to the training data (scanner, sequence, slice orientation, preprocessing).
Heatmaps show where the model looked, not where the tumour is, and a segmentation is a model output, not a radiologist's outline. Confidence is not the probability that a patient has a disease.
The system must not be used for diagnosis or treatment decisions. A qualified radiologist or clinician must review every scan. Patient data should be synthetic or properly anonymised and handled with permission."""),
]


def _tok(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", s.lower()):
        if len(w) < 2 or w in _STOP:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith("es") and not w.endswith("ses"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text) if x.strip()]


def _chunk_text(text, size=900, overlap=150):
    sents = _sentences(re.sub(r"\s+", " ", text).strip())
    chunks, cur = [], ""
    for s_ in sents:
        if cur and len(cur) + len(s_) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:]
            cur = (tail[tail.find(" ") + 1:] if " " in tail else tail) + " " + s_
        else:
            cur = (cur + " " + s_).strip()
    if cur:
        chunks.append(cur)
    return chunks


def _read_doc(path):
    """-> list of (page_number, text)."""
    low = path.lower()
    if low.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pypdf is not installed (run: %pip install -q pypdf)")
        r = PdfReader(path)
        return [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(r.pages)]
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pages, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > 2400:
            pages.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip()
    if cur:
        pages.append(cur)
    return [(i + 1, t) for i, t in enumerate(pages)]


def _rag_dir():
    d = os.path.join(DRIVE_FOLDER, RAG_DIR_NAME) if os.path.isdir(DRIVE_FOLDER) else os.path.join(os.getcwd(), RAG_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


class _BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [_tok(d) for d in docs]
        self.len = np.array([len(t) for t in self.toks], dtype=np.float64)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        self.post = {}
        for i, t in enumerate(self.toks):
            for w in set(t):
                self.post.setdefault(w, []).append(i)
        self.tf = [dict() for _ in self.toks]
        for i, t in enumerate(self.toks):
            for w in t:
                self.tf[i][w] = self.tf[i].get(w, 0) + 1
        n = len(self.toks)
        self.idf = {w: float(np.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))) for w, p in self.post.items()}

    def scores(self, query):
        sc = np.zeros(len(self.toks))
        for w in set(_tok(query)):
            for i in self.post.get(w, []):
                f = self.tf[i][w]
                sc[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sc


def _embedder(kind="auto"):
    """Sentence-transformers if available (runs locally), else Gemini embeddings, else none (BM25 only)."""
    if kind == "none":
        return None, "BM25 keyword search only"
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(RAG_EMBED_MODELS[0])
        return (lambda texts: np.asarray(m.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)), \
            f"BM25 + {RAG_EMBED_MODELS[0]} embeddings (reciprocal rank fusion)"
    except Exception:
        pass
    if S.gemini_client is not None:
        def gem(texts):
            out = []
            for i in range(0, len(texts), 50):
                r = S.gemini_client.models.embed_content(model=RAG_EMBED_MODELS[1], contents=texts[i:i + 50])
                out += [e.values for e in r.embeddings]
            a = np.asarray(out, dtype=np.float32)
            return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        try:
            gem(["test"])
            return gem, f"BM25 + {RAG_EMBED_MODELS[1]} embeddings (reciprocal rank fusion)"
        except Exception:
            pass
    return None, "BM25 keyword search only"


def build_rag_index(embeddings="auto", verbose=True):
    """Read every document, split into page-level chunks and index them (keyword + vector)."""
    t0 = time.perf_counter()
    S.rag_status = {"state": "building"}
    chunks, docs = [], []
    for title, text in HANDBOOK:
        for c in _chunk_text(text):
            chunks.append({"doc": "NeuroNova handbook (built in)", "section": title, "page": None, "text": c,
                           "builtin": True})
    docs.append({"name": "NeuroNova handbook (built in)", "pages": len(HANDBOOK), "kind": "built in",
                 "chunks": len(chunks), "error": None})
    d = _rag_dir()
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(RAG_EXT) or f.startswith("."):
            continue
        p = os.path.join(d, f)
        n0 = len(chunks)
        try:
            pages = _read_doc(p)
            for pno, text in pages:
                for c in _chunk_text(text):
                    if len(c) > 40:
                        chunks.append({"doc": f, "section": None, "page": pno, "text": c, "builtin": False})
            docs.append({"name": f, "pages": len(pages), "kind": f.rsplit(".", 1)[-1].upper(),
                         "chunks": len(chunks) - n0, "error": None,
                         "size_kb": round(os.path.getsize(p) / 1024)})
        except Exception as exc:
            docs.append({"name": f, "pages": 0, "kind": f.rsplit(".", 1)[-1].upper(), "chunks": 0,
                         "error": f"{type(exc).__name__}: {exc}"})
        if verbose:
            print(f"  indexed {f}: {docs[-1]['pages']} pages, {docs[-1]['chunks']} passages"
                  + (f" ({docs[-1]['error']})" if docs[-1]["error"] else ""))
    bm = _BM25([(c["section"] or "") + " " + c["text"] for c in chunks])
    emb, method = _embedder(embeddings)
    vecs = None
    if emb is not None:
        try:
            vecs = emb([c["text"] for c in chunks])
        except Exception as exc:
            vecs, method = None, f"BM25 keyword search only (embeddings failed: {type(exc).__name__})"
    S.rag = {"chunks": chunks, "docs": docs, "bm25": bm, "vecs": vecs, "embed": emb if vecs is not None else None,
             "method": method, "folder": d}
    S.rag_status = {"state": "ready", "chunks": len(chunks), "docs": len(docs), "method": method,
                    "seconds": round(time.perf_counter() - t0, 1)}
    if verbose:
        print(f"  Knowledge base ready: {len(docs)} documents, {len(chunks)} passages · {method}")
    return S.rag


def rag_search(query, k=5):
    if not S.rag:
        build_rag_index(verbose=False)
    R = S.rag
    bm = R["bm25"].scores(query)
    order_b = np.argsort(-bm)
    rrf = np.zeros(len(bm))
    rank_b = np.empty(len(bm), dtype=int)
    rank_b[order_b] = np.arange(len(bm))
    rrf += np.where(bm > 0, 1.0 / (60 + rank_b), 0.0)
    dense = None
    if R["vecs"] is not None:
        try:
            q = R["embed"]([query])[0]
            dense = R["vecs"] @ q
            order_d = np.argsort(-dense)
            rank_d = np.empty(len(dense), dtype=int)
            rank_d[order_d] = np.arange(len(dense))
            rrf += 1.0 / (60 + rank_d)
        except Exception:
            dense = None
    top = np.argsort(-rrf)[:k]
    out = []
    for i in top:
        if rrf[i] <= 0:
            continue
        c = R["chunks"][i]
        out.append({**{kk: c[kk] for kk in ("doc", "section", "page", "text", "builtin")},
                    "bm25": float(bm[i]), "dense": None if dense is None else float(dense[i]), "score": float(rrf[i])})
    return out


def _supported(hits, query):
    if not hits:
        return False
    qt = set(_tok(query))
    best_overlap = max(len(qt & set(_tok(h["text"]))) / max(1, len(qt)) for h in hits)
    dense_ok = any(h["dense"] is not None and h["dense"] >= 0.45 for h in hits)
    return best_overlap >= 0.34 or dense_ok


def _verify(answer, hits):
    """Check each cited sentence against the passages it cites (word overlap of the content words)."""
    out = []
    for s_ in _sentences(answer):
        cites = [int(x) for x in re.findall(r"\[(\d+)\]", s_)]
        words = set(_tok(re.sub(r"\[\d+\]", "", s_)))
        if not cites:
            out.append({"sentence": s_, "cites": [], "support": None})
            continue
        pool = set()
        for n in cites:
            if 1 <= n <= len(hits):
                pool |= set(_tok(hits[n - 1]["text"]))
        sup = len(words & pool) / max(1, len(words))
        out.append({"sentence": s_, "cites": cites, "support": float(sup),
                    "invalid_cite": any(not (1 <= n <= len(hits)) for n in cites)})
    return out


def rag_answer(question, case=None, k=5):
    """Evidence-grounded answer: retrieved passages, a cited answer, a separate model explanation and checks."""
    t0 = time.perf_counter()
    hits = rag_search(question, k)
    supported = _supported(hits, question)
    facts = ""
    if case is not None and case.get("results"):
        r = case["results"]
        h = (r.get("hybrid") or {}).get("hybrid") or r.get("classify") or {}
        dec = (r.get("report") or {}).get("decision") or {}
        sg = r.get("segment") or {}
        facts = (f"Prediction {h.get('prediction')} ({h.get('confidence', 0):.1f}%); decision {dec.get('status', 'n/a')}"
                 + (f"; segmented area {sg.get('area_px')} px" if sg.get("detected") else ""))
    used_llm = False
    answer, explanation = None, None
    if S.gemini_client is not None and S.gemini_types is not None and hits:
        src = "\n\n".join(f"[{i + 1}] {h['doc']}" + (f", page {h['page']}" if h["page"] else "")
                          + (f", section '{h['section']}'" if h["section"] else "") + f":\n{h['text']}"
                          for i, h in enumerate(hits))
        prompt = f"""You answer questions using ONLY the numbered source passages below.
Rules:
- Every sentence of "answer" must end with the citation(s) of the passage(s) it comes from, like [1] or [2][3].
- Do not use outside knowledge in "answer". If the passages do not contain the answer, set "supported" to false
  and write "The available sources do not answer this question." as the answer.
- "explanation" may add general background in plain words, WITHOUT citations, clearly separate from the sources.
  Keep it under 80 words, or leave it empty.
- This is educational information, not medical advice.
Return JSON only: {{"answer": str, "supported": bool, "explanation": str}}

SOURCES:
{src}

{('CASE FACTS FROM NEURONOVA (not a source, do not cite): ' + facts) if facts else ''}

QUESTION: {question}"""
        try:
            resp = S.gemini_client.models.generate_content(
                model=S.gemini_model, contents=prompt,
                config=S.gemini_types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"))
            txt = (getattr(resp, "text", None) or "").strip()
            m = re.search(r"\{.*\}", txt, re.S)
            data = json.loads(m.group(0) if m else txt)
            answer = str(data.get("answer") or "").strip()
            explanation = str(data.get("explanation") or "").strip() or None
            supported = bool(data.get("supported", supported)) and bool(answer)
            used_llm = True
        except Exception:
            answer = None
    if answer is None:
        if supported:
            qt = set(_tok(question))
            need = 1 if len(qt) < 2 else int(np.ceil(0.5 * len(qt)))
            top_bm = max([h["bm25"] for h in hits] + [1e-9])
            cand, seen = [], set()
            for i, h in enumerate(hits):
                if h["bm25"] < 0.35 * top_bm and not (h["dense"] is not None and h["dense"] >= 0.45):
                    continue
                for s_ in _sentences(h["text"]):
                    key = re.sub(r"\W+", " ", s_.lower()).strip()
                    ov = len(qt & set(_tok(s_)))
                    if ov >= need and key not in seen and len(s_) > 25:
                        seen.add(key)
                        cand.append((ov / max(1, len(qt)) + 0.3 * h["bm25"] / top_bm, i, s_))
            cand.sort(key=lambda x: -x[0])
            picked = sorted(cand[:3], key=lambda x: x[1])
            answer = " ".join(f"{s_.rstrip().rstrip('.!?')} [{i + 1}]." for _, i, s_ in picked) if picked else \
                "The available sources do not answer this question."
            supported = bool(picked)
        else:
            answer = "The available sources do not answer this question."
    if not supported:
        answer = "The available sources do not answer this question." + (
            "" if answer.startswith("The available sources") else "")
    checks = _verify(answer, hits) if supported else []
    weak = [c for c in checks if c["support"] is not None and (c["support"] < 0.3 or c.get("invalid_cite"))]
    cited = sorted({n for c in checks for n in c["cites"] if 1 <= n <= len(hits)})
    return {"question": question, "answer": answer, "supported": supported, "explanation": explanation,
            "sources": [{**h, "n": i + 1, "cited": (i + 1) in cited} for i, h in enumerate(hits)],
            "verification": checks, "weakly_supported": len(weak),
            "method": S.rag["method"] if S.rag else None,
            "writer": f"Gemini {S.gemini_model}" if used_llm else "extractive (no language model)",
            "ms": _ms(t0)}


def show_rag(res):
    print("QUESTION:", res["question"])
    print("\nANSWER FROM THE SOURCES" + ("" if res["supported"] else " (not supported)") + f" · {res['writer']}:")
    print(" ", res["answer"])
    if res.get("explanation"):
        print("\nMODEL EXPLANATION (not from the sources):\n ", res["explanation"])
    print("\nSOURCES:")
    for s_ in res["sources"]:
        where = s_["doc"] + (f", page {s_['page']}" if s_["page"] else "") + (f" — {s_['section']}" if s_["section"] else "")
        print(f"  [{s_['n']}] {'(cited) ' if s_['cited'] else ''}{where}")
        print(f"      {s_['text'][:220]}{'...' if len(s_['text']) > 220 else ''}")
    if res["verification"]:
        ok = sum(1 for c in res["verification"] if c["support"] is not None and c["support"] >= 0.3)
        print(f"\nCHECK: {ok} of {len([c for c in res['verification'] if c['cites']])} cited sentences "
              f"share most of their key words with the passages they cite.")
    print(f"Retrieval: {res['method']}")


# ================================================================
# TOOL 14 — MULTI-AGENT RESEARCH WORKFLOW
#   Image agent -> XAI agent -> Report-text agent -> Evidence agent ->
#   Validation agent -> Writer agent -> Safety check.
#   Each agent has a fixed input/output schema, every tool call is
#   logged, a failing agent does not stop the others, and the final
#   safety check removes diagnostic wording. Agents never turn a model
#   output into a confirmed diagnosis.
# ================================================================
AGENT_SCHEMAS = {
    "image": {"prediction": str, "confidence": float, "calibrated_confidence": float, "decision": str,
              "models": dict, "quality": str, "looks_like_mri": str},
    "xai": {"gradcam_percent": float, "occlusion_agreement": str, "xai_consensus": str, "segmentation": dict},
    "report_text": {"has_text": bool, "mentions": list, "negated_tumour": bool, "sizes_mm": list,
                    "laterality": list, "locations": list, "impression": str},
    "evidence": {"queries": list, "passages": list},
    "validation": {"findings": list},
    "writer": {"markdown": str, "sections": dict},
    "safety": {"changes": list, "passed": bool},
}
_TYPE_PATTERNS = {"glioma": r"gliom|glioblastom|astrocytom|oligodendrogliom|\bgbm\b",
                  "meningioma": r"meningiom",
                  "pituitary": r"pituitar|adenom|macroadenom|microadenom|sellar",
                  "notumor": r"no (evidence of )?(a )?(tumou?r|mass|lesion)|unremarkable|normal (mri|study|brain)"}
_LOCATIONS = ["frontal", "parietal", "temporal", "occipital", "cerebellar", "cerebellum", "brainstem", "thalamic",
              "sellar", "suprasellar", "parasagittal", "convexity", "falx", "intraventricular", "periventricular"]
_DIAGNOSTIC = [(r"\b(is|was|has been) diagnosed with\b", r"is predicted by the model to show"),
               (r"\bconfirm(s|ed)? (the|a|this) diagnosis\b", "supports the model's prediction"),
               (r"\bconfirm(s|ed)? (the )?presence of\b", "is consistent with"),
               (r"\bdefinitely\b", "possibly"), (r"\bcertainly\b", "possibly"),
               (r"\byou have (a |an )?", r"the model suggests \1"),
               (r"\bthe patient has (a |an )?", r"the image may show \1"),
               (r"\b(should|must) (start|begin|undergo|receive|have)\b", r"could ask the care team whether to \2"),
               (r"\bdiagnosis:", "model output:")]
DISCLAIMER = ("This is an AI-generated research summary from a student project. It is not a diagnosis and must be "
              "reviewed by a qualified radiologist or clinician.")


def _check_schema(name, out):
    problems = []
    for k, t in AGENT_SCHEMAS[name].items():
        if k not in out:
            problems.append(f"missing '{k}'")
        elif out[k] is not None and not isinstance(out[k], t) and not (t is float and isinstance(out[k], int)):
            problems.append(f"'{k}' should be {t.__name__}")
    return problems


def _agent_image(case):
    r = case["results"]
    h = r.get("hybrid") or {}
    hy = h.get("hybrid") or r.get("classify") or {}
    dec = (r.get("report") or {}).get("decision") or {}
    unc = r.get("uncertainty") or {}
    models = {k: {"prediction": v["prediction"], "confidence": v["confidence"]}
              for k, v in h.items() if isinstance(v, dict) and "prediction" in v and k in ("efficientnet", "inception", "densenet")}
    return {"prediction": hy.get("prediction"), "confidence": float(hy.get("confidence", 0.0)),
            "calibrated_confidence": float((h.get("calibrated") or {}).get("confidence", hy.get("confidence", 0.0))),
            "decision": dec.get("status", "not run"), "models": models,
            "quality": (unc.get("quality") or {}).get("verdict", "not run"),
            "looks_like_mri": (unc.get("ood") or {}).get("verdict", "not run"),
            "reasons": dec.get("reasons", []), "epistemic": (unc.get("hybrid") or {}).get("epistemic")}


def _agent_xai(case):
    r = case["results"]
    g, o, x, sg = r.get("gradcam") or {}, r.get("occlusion") or {}, r.get("xai") or {}, r.get("segment") or {}
    seg = {}
    if sg.get("available"):
        seg = {"detected": sg["detected"], "area_px": sg.get("area_px"), "area_percent_image": sg.get("area_percent_image"),
               "location": sg.get("location"), "equivalent_diameter_px": sg.get("equivalent_diameter_px"),
               "area_mm2": sg.get("area_mm2"), "consistent_with_classifier": sg.get("consistent_with_classifier"),
               "gradcam_peak_inside": (sg.get("gradcam_vs_mask") or {}).get("peak_inside")}
    return {"gradcam_percent": float(g.get("activation_percent", 0.0)) if g else None,
            "occlusion_agreement": o.get("agreement_level", "not run"),
            "xai_consensus": x.get("strength", "not run"), "segmentation": seg,
            "occlusion_iou": o.get("gradcam_iou"), "xai_mean_iou": x.get("mean_pairwise_iou")}


def _agent_report_text(text):
    text = (text or "").strip()
    low = text.lower()
    out = {"has_text": bool(text), "mentions": [], "negated_tumour": False, "sizes_mm": [], "laterality": [],
           "locations": [], "impression": ""}
    if not text:
        return out
    for lab, pat in _TYPE_PATTERNS.items():
        if lab != "notumor" and re.search(pat, low):
            out["mentions"].append(lab)
    out["negated_tumour"] = bool(re.search(_TYPE_PATTERNS["notumor"], low))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)(?:\s*(?:x|×|by)\s*(\d+(?:\.\d+)?))?\s*(mm|cm)\b", low):
        f = 10.0 if m.group(4) == "cm" else 1.0
        out["sizes_mm"].append([float(v) * f for v in m.groups()[:3] if v])
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", low):
        f = 10.0 if m.group(2) == "cm" else 1.0
        v = float(m.group(1)) * f
        if not any(v in sz for sz in out["sizes_mm"]):
            out["sizes_mm"].append([v])
    out["laterality"] = sorted({w for w in ("left", "right", "bilateral", "midline") if re.search(rf"\b{w}\b", low)})
    out["locations"] = [w for w in _LOCATIONS if w in low]
    m = re.search(r"(impression|conclusion)s?\s*[:\-]\s*(.+?)(\n\n|$)", text, re.I | re.S)
    out["impression"] = (m.group(2).strip()[:400] if m else _sentences(text)[-1][:400] if _sentences(text) else "")
    return out


def _agent_evidence(img, rep):
    names = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary tumour", "notumor": "a normal brain MRI"}
    pred = img.get("prediction") or ""
    queries = []
    if pred:
        queries.append(f"What does {names.get(pred, pred)} look like on MRI?")
    for mtn in rep.get("mentions", []):
        if mtn != pred:
            queries.append(f"How is {names.get(mtn, mtn)} distinguished on MRI?")
    queries.append("What are the limitations of the AI model and how should results be used safely?")
    passages, seen = [], set()
    for q in queries[:3]:
        for h in rag_search(q, 3):
            key = (h["doc"], h["page"], h["text"][:60])
            if key in seen:
                continue
            seen.add(key)
            passages.append({"query": q, "doc": h["doc"], "page": h["page"], "section": h["section"],
                             "text": h["text"][:500], "score": h["score"]})
    return {"queries": queries[:3], "passages": passages[:6]}


def _evidence_snippet(text, query, limit=220):
    """Whole sentences from a passage, the ones that best match the question, never starting mid-sentence."""
    sents = _sentences(re.sub(r"\s+", " ", text or "").strip())
    if len(sents) > 1 and not re.match(r"[A-Z0-9(\"']", sents[0]):
        sents = sents[1:]                                   # chunk overlap can start mid-sentence
    if not sents:
        return ""
    q = set(_tok(query))
    best = max(range(len(sents)), key=lambda i: (len(q & set(_tok(sents[i]))), -i))
    out = sents[best]
    for nxt in sents[best + 1:]:
        if len(out) + len(nxt) + 1 > limit:
            break
        out += " " + nxt
    return out if len(out) <= limit + 60 else out[:limit].rsplit(" ", 1)[0] + "..."


def _agent_validate(case, img, xai, rep, ev):
    F = []

    def add(kind, msg, evidence=None):
        F.append({"type": kind, "message": msg, "evidence": evidence})
    pred = img.get("prediction")
    tumour_pred = pred not in (None, "notumor")
    if img.get("decision") == "abstain":
        add("warning", "The reject option withheld the model's class for this image.", "; ".join(img.get("reasons", [])))
    if img.get("looks_like_mri") == "not_mri":
        add("contradiction", "The input validation says the image does not look like a brain MRI, yet a class was predicted.")
    ms = img.get("models") or {}
    preds = {v["prediction"] for v in ms.values()}
    if len(preds) > 1:
        mn = {"efficientnet": "EfficientNetV2-B0", "inception": "InceptionV3", "densenet": "DenseNet121"}
        add("warning", "The CNNs disagree: " + ", ".join(f"{mn.get(k, k)} {v['prediction']}" for k, v in ms.items()) + ".")
    sg = xai.get("segmentation") or {}
    if sg and sg.get("consistent_with_classifier") is False:
        add("contradiction", "The U-Net segmentation and the classifier disagree about whether a tumour is present.")
    if sg and sg.get("detected") and sg.get("gradcam_peak_inside") is False:
        add("warning", "Grad-CAM's hottest point lies outside the segmented region.")
    if rep.get("has_text"):
        if rep.get("negated_tumour") and tumour_pred and not rep.get("mentions"):
            add("contradiction", f"The report text says there is no tumour, but the model predicts '{pred}'.", rep.get("impression"))
        if rep.get("mentions") and pred and pred not in rep["mentions"]:
            add("contradiction", f"The report mentions {', '.join(rep['mentions'])}, but the model predicts '{pred}'.",
                rep.get("impression"))
        if rep.get("mentions") and pred in rep["mentions"]:
            add("ok", f"The report text and the model both point to {pred}.")
        sizes = [max(sz) for sz in rep.get("sizes_mm", []) if sz]
        if sizes and sg.get("detected"):
            if sg.get("area_mm2"):
                dia = 2 * np.sqrt(sg["area_mm2"] / np.pi)
                rel = abs(dia - max(sizes)) / max(sizes)
                add("ok" if rel <= 0.5 else "contradiction",
                    f"Report size {max(sizes):.0f} mm vs segmented diameter about {dia:.0f} mm "
                    f"({'consistent' if rel <= 0.5 else 'more than 50% apart'}).")
            else:
                add("missing", "The report gives a size, but this image has no pixel spacing, so the segmented size "
                               "cannot be converted to millimetres (use a DICOM file or a 3D scan).")
        if rep.get("laterality"):
            add("missing", f"The report says {', '.join(rep['laterality'])}; left/right cannot be checked from a 2D "
                           "image without orientation information.")
    else:
        add("missing", "No radiology report text was supplied, so the model cannot be checked against one.")
    if not sg:
        add("missing", "No segmentation (train the Tool 10 U-Net) — tumour size and position were not measured.")
    if (reliability_settings().get("source") == "default"):
        add("missing", "The reject threshold is a default; run the model evaluation (Tool 8D) to calibrate it.")
    if not ev.get("passages") or all(p["doc"].startswith("NeuroNova handbook") for p in ev.get("passages", [])):
        add("missing", "No trusted medical documents were found for this question; only the built-in handbook was used.")
    return {"findings": F}


def _agent_writer(case, img, xai, rep, ev, val):
    names = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary tumour", "notumor": "no tumour"}
    pred = img.get("prediction")
    sec = {}
    sec["summary"] = (f"The hybrid model's output is **{names.get(pred, pred)}** with {img['confidence']:.1f}% confidence "
                      f"({img['calibrated_confidence']:.1f}% after calibration). "
                      + ("The reject option **withheld** this class." if img.get("decision") == "abstain"
                         else "The result passed the reject option." if img.get("decision") == "report" else ""))
    mn = {"efficientnet": "EfficientNetV2-B0", "inception": "InceptionV3", "densenet": "DenseNet121"}
    ml = "; ".join(f"{mn.get(k, k)}: {v['prediction']} {v['confidence']:.1f}%"
                   for k, v in sorted((img.get("models") or {}).items(), key=lambda kv: list(mn).index(kv[0]) if kv[0] in mn else 9))
    sec["image_analysis"] = (f"Models: {ml or 'n/a'}. Input quality: {img.get('quality')}; brain-MRI check: "
                             f"{img.get('looks_like_mri')}." + (f" Epistemic uncertainty {img['epistemic']:.3f}." if img.get("epistemic") is not None else ""))
    sg = xai.get("segmentation") or {}
    sec["explainability"] = (f"Grad-CAM marks {xai['gradcam_percent']:.1f}% of the image; occlusion agreement: "
                             f"{xai.get('occlusion_agreement')}; explanation-method consensus: {xai.get('xai_consensus')}."
                             if xai.get("gradcam_percent") is not None else "Explanations were not run.")
    if sg:
        sec["explainability"] += (f" U-Net: {'region of ' + format(sg['area_px'], ',') + ' px (' + format(sg['area_percent_image'], '.2f') + '% of the image) in the ' + sg['location'] if sg.get('detected') else 'no tumour region'}.")
    sec["report_text"] = ("No report text was supplied." if not rep.get("has_text") else
                          f"Mentions: {', '.join(rep['mentions']) or 'none'}; no-tumour wording: {'yes' if rep['negated_tumour'] else 'no'}; "
                          f"sizes: {', '.join('x'.join(f'{v:.0f}' for v in sz) + ' mm' for sz in rep['sizes_mm']) or 'none'}; "
                          f"locations: {', '.join(rep['locations']) or 'none'}. Impression: \"{rep['impression']}\"")
    sec["evidence"] = "\n".join(f"- [{i + 1}] {p['doc']}" + (f", page {p['page']}" if p['page'] else "")
                                + f": \"{_evidence_snippet(p['text'], p.get('query', ''))}\"" for i, p in enumerate(ev.get("passages", [])[:4])) or "- none"
    sec["validation"] = "\n".join(f"- **{f['type']}**: {f['message']}" for f in val.get("findings", [])) or "- no issues"
    narrative = None
    if S.gemini_client is not None and S.gemini_types is not None:
        facts = json.dumps({"image": img, "xai": xai, "report_text": rep,
                            "validation": val.get("findings", []), "evidence": [p["text"][:300] for p in ev.get("passages", [])[:4]]},
                           default=str)[:9000]
        prompt = ("Write a 120-word research summary of this AI analysis for a supervisor. Use ONLY the facts in the JSON. "
                  "Do not diagnose, do not recommend treatment, use hedged wording ('the model suggests'). "
                  "Mention contradictions and missing information. Plain text, no headings.\n\nFACTS:\n" + facts)
        try:
            resp = S.gemini_client.models.generate_content(model=S.gemini_model, contents=prompt,
                                                           config=S.gemini_types.GenerateContentConfig(temperature=0.2))
            narrative = (getattr(resp, "text", None) or "").strip() or None
        except Exception:
            narrative = None
    if narrative:
        sec["narrative"] = narrative
    md_ = [f"# NeuroNova research report · {case['case_id']}",
           f"*File: {case.get('filename')} · generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}*", ""]
    for key, title in (("summary", "Summary"), ("narrative", "Narrative (Gemini, from the facts below)"),
                       ("image_analysis", "Image analysis"), ("explainability", "Explainability"),
                       ("report_text", "Radiology report text"), ("evidence", "Evidence from the knowledge base"),
                       ("validation", "Validation: contradictions and missing information")):
        if key in sec:
            md_ += [f"## {title}", sec[key], ""]
    md_ += ["## Safety", DISCLAIMER]
    return {"markdown": "\n".join(md_), "sections": sec}


def _agent_safety(writer):
    text = writer["markdown"]
    changes = []
    for pat, repl in _DIAGNOSTIC:
        for m in re.finditer(pat, text, re.I):
            changes.append({"found": m.group(0), "replaced_with": m.expand(repl)})
        text = re.sub(pat, repl, text, flags=re.I)
    if changes:
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text, flags=re.M)
    if DISCLAIMER not in text:
        text += "\n\n" + DISCLAIMER
        changes.append({"found": "missing disclaimer", "replaced_with": "disclaimer added"})
    writer["markdown"] = text
    return {"changes": changes, "passed": True}


def run_agents(case, report_text="", verbose=False):
    """Run the controlled multi-agent workflow on an analysed case; every call is logged."""
    t_all = time.perf_counter()
    if "report" not in case["results"]:
        raise ValueError("Run the analysis pipeline on this case first.")
    log, outs = [], {}
    plan = [
        ("image", "Image agent", "reads the classification pipeline (Tools 1, 3, 8A, reject option)", lambda: _agent_image(case)),
        ("xai", "XAI agent", "collects Grad-CAM, occlusion, the XAI lab and the segmentation", lambda: _agent_xai(case)),
        ("report_text", "Report-text agent", "extracts tumour types, sizes and locations from the report text",
         lambda: _agent_report_text(report_text)),
        ("evidence", "Evidence agent", "retrieves cited passages from the knowledge base (Tool 13)",
         lambda: _agent_evidence(outs.get("image", {}), outs.get("report_text", {}))),
        ("validation", "Validation agent", "looks for contradictions and missing information",
         lambda: _agent_validate(case, outs.get("image", {}), outs.get("xai", {}), outs.get("report_text", {}),
                                 outs.get("evidence", {}))),
        ("writer", "Writer agent", "assembles the structured research report",
         lambda: _agent_writer(case, outs.get("image", {}), outs.get("xai", {}), outs.get("report_text", {}),
                               outs.get("evidence", {}), outs.get("validation", {}))),
        ("safety", "Safety check", "removes diagnostic wording and adds the disclaimer",
         lambda: _agent_safety(outs["writer"]) if "writer" in outs else {"changes": [], "passed": False}),
    ]
    if S.rag is None:
        try:
            build_rag_index(verbose=False)
        except Exception:
            pass
    for key, name, role, fn in plan:
        t0 = time.perf_counter()
        entry = {"agent": name, "key": key, "role": role, "status": "ok", "error": None}
        try:
            out = fn()
            problems = _check_schema(key, out)
            if problems:
                entry["status"] = "schema"
                entry["error"] = "; ".join(problems)
            outs[key] = out
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            outs[key] = {}
        entry["ms"] = _ms(t0)
        o = outs.get(key) or {}
        entry["output"] = {
            "image": lambda: f"{o.get('prediction')} {o.get('confidence', 0):.1f}% · decision {o.get('decision')}",
            "xai": lambda: f"Grad-CAM {o.get('gradcam_percent') or 0:.1f}% · occlusion {o.get('occlusion_agreement')} · consensus {o.get('xai_consensus')}",
            "report_text": lambda: (f"mentions {', '.join(o.get('mentions') or []) or 'none'} · sizes {len(o.get('sizes_mm') or [])}"
                                    if o.get("has_text") else "no report text supplied"),
            "evidence": lambda: f"{len(o.get('passages') or [])} passages for {len(o.get('queries') or [])} queries",
            "validation": lambda: " · ".join(f"{n_} {lab}{'' if n_ == 1 else 's'}" for n_, lab in (
                (sum(1 for f in o.get('findings', []) if f['type'] == 'contradiction'), "contradiction"),
                (sum(1 for f in o.get('findings', []) if f['type'] == 'missing'), "missing item"),
                (sum(1 for f in o.get('findings', []) if f['type'] == 'warning'), "warning"))),
            "writer": lambda: f"{len((o.get('markdown') or '').split())} words in {len(o.get('sections') or {})} sections",
            "safety": lambda: f"{len(o.get('changes') or [])} edits",
        }[key]() if o else "no output"
        log.append(entry)
        if verbose:
            print(f"  {'✓' if entry['status'] == 'ok' else '!'} {name:<18} {entry['ms'] / 1000:5.2f} s  {entry['output']}"
                  + (f"  [{entry['error']}]" if entry["error"] else ""))
    res = {"case_id": case["case_id"], "steps": log, "outputs": {k: v for k, v in outs.items() if k != "writer"},
           "report_markdown": (outs.get("writer") or {}).get("markdown", ""),
           "findings": (outs.get("validation") or {}).get("findings", []),
           "safety": outs.get("safety", {}), "ms": _ms(t_all)}
    case["agents"] = res
    return res


def show_agents(res):
    from IPython.display import Markdown, display
    print(f"TOOL 14 — multi-agent workflow for {res['case_id']} ({res['ms'] / 1000:.1f} s)")
    for st in res["steps"]:
        print(f"  {'OK ' if st['status'] == 'ok' else st['status'].upper():6} {st['agent']:<18} {st['output']}")
    display(Markdown(res["report_markdown"]))


# ================================================================
# CASE MANAGEMENT APP — login and roles, SQLite records, persistent case
# history, doctor review and correction, audit log, case comparison and
# PDF reports. Records live in brain_tumor_models/neuronova_data on
# Drive, so they survive a new Colab session. Use synthetic patients
# unless you are authorised to handle real patient data.
# ================================================================
import sqlite3
import hashlib
import secrets
import contextvars
import gzip

DATA_DIR_NAME = "neuronova_data"
ROLES = ("admin", "doctor", "researcher")
SESSION_HOURS = 12
_CURRENT_USER = contextvars.ContextVar("neuronova_user", default=None)
_LOCAL_USER = {"username": "local", "name": "Local user", "role": "admin"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, name TEXT, role TEXT NOT NULL, salt TEXT, pw_hash TEXT,
    active INTEGER DEFAULT 1, created_at TEXT, last_login TEXT);
CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, username TEXT, created_at TEXT, expires_at REAL);
CREATE TABLE IF NOT EXISTS patients (patient_id TEXT PRIMARY KEY, name TEXT, age INTEGER, sex TEXT, history TEXT,
    synthetic INTEGER DEFAULT 1, created_at TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, patient_id TEXT, filename TEXT, source TEXT, created_at TEXT,
    created_by TEXT, prediction TEXT, confidence REAL, calibrated REAL, decision TEXT, seg_area REAL, reliability TEXT,
    steps TEXT DEFAULT '', status TEXT DEFAULT 'analysed', review_verdict TEXT, review_label TEXT, review_note TEXT,
    reviewed_by TEXT, reviewed_at TEXT, thumb TEXT, ground_truth TEXT, q_bright REAL, q_contrast REAL, q_sharp REAL,
    ood_ratio REAL);
CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, reviewer TEXT, verdict TEXT,
    label TEXT, note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, username TEXT, role TEXT, action TEXT,
    target TEXT, detail TEXT);
"""
SYNTHETIC_PATIENTS = [("DEMO Patient 01", 54, "F", "Synthetic record: headaches for 3 months."),
                      ("DEMO Patient 02", 37, "M", "Synthetic record: one seizure, no previous history."),
                      ("DEMO Patient 03", 61, "F", "Synthetic record: blurred vision, hormone tests pending."),
                      ("DEMO Patient 04", 45, "M", "Synthetic record: routine follow-up scan."),
                      ("DEMO Patient 05", 29, "F", "Synthetic record: dizziness and nausea.")]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Store:
    """SQLite metadata on the local disk, copied to Drive after every change; case files go straight to Drive."""

    def __init__(self, data_dir):
        self.dir = data_dir
        os.makedirs(os.path.join(data_dir, "cases"), exist_ok=True)
        self.backup = os.path.join(data_dir, "neuronova.db")
        local_dir = "/content" if os.path.isdir("/content") else data_dir
        self.path = os.path.join(local_dir, ".neuronova_local.db") if local_dir != data_dir else self.backup
        if self.path != self.backup and os.path.exists(self.backup):
            shutil.copy(self.backup, self.path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(cases)").fetchall()}
        for col in ("q_bright", "q_contrast", "q_sharp", "ood_ratio"):
            if col not in have:
                self.conn.execute(f"ALTER TABLE cases ADD COLUMN {col} REAL")
        self.conn.commit()
        self._dirty = threading.Event()
        if self.path != self.backup:
            threading.Thread(target=self._syncer, daemon=True).start()

    def q(self, sql, args=(), one=False):
        with self.lock:
            cur = self.conn.execute(sql, args)
            rows = [dict(r) for r in cur.fetchall()]
            if not sql.lstrip().upper().startswith("SELECT"):
                self.conn.commit()
                self._dirty.set()
        return (rows[0] if rows else None) if one else rows

    def _syncer(self):
        while True:
            self._dirty.wait()
            time.sleep(2)
            self._dirty.clear()
            try:
                with self.lock:
                    bk = sqlite3.connect(self.backup + ".tmp")
                    self.conn.backup(bk)
                    bk.close()
                os.replace(self.backup + ".tmp", self.backup)
            except Exception:
                pass

    def case_file(self, case_id, ext):
        return os.path.join(self.dir, "cases", f"{case_id}.{ext}")


def _hash_pw(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    return salt, hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt), 200_000).hex()


def _setup_store(auth=True):
    data_dir = os.path.join(DRIVE_FOLDER if os.path.isdir(DRIVE_FOLDER) else os.getcwd(), DATA_DIR_NAME)
    S.store = _Store(data_dir)
    S.auth_enabled = bool(auth)
    st = S.store
    if not st.q("SELECT COUNT(*) AS n FROM users", one=True)["n"]:
        lines = ["NeuroNova demo accounts (created " + _now() + "). Change them in the Admin panel.", ""]
        for user, name, role in (("admin", "Project admin", "admin"), ("doctor", "Dr. Demo", "doctor"),
                                 ("researcher", "Research student", "researcher")):
            pw = secrets.token_urlsafe(6)
            salt, h = _hash_pw(pw)
            st.q("INSERT INTO users (username, name, role, salt, pw_hash, active, created_at) VALUES (?,?,?,?,?,1,?)",
                 (user, name, role, salt, h, _now()))
            lines.append(f"{role:<11} username: {user:<11} password: {pw}")
        with open(os.path.join(data_dir, "demo_logins.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        if auth:
            print("\n  " + "\n  ".join(lines))
            print(f"  (also saved in {os.path.join(data_dir, 'demo_logins.txt')})")
    elif auth:
        f = os.path.join(data_dir, "demo_logins.txt")
        if os.path.exists(f):
            print("\n  " + "\n  ".join(open(f).read().strip().split("\n")))
            print("  (passwords changed in the Admin panel are not updated in this file)")
        else:
            print("  Sign-in           : use the accounts created earlier (Admin panel)")
    if not st.q("SELECT COUNT(*) AS n FROM patients", one=True)["n"]:
        for i, (name, age, sex, hist) in enumerate(SYNTHETIC_PATIENTS):
            st.q("INSERT INTO patients VALUES (?,?,?,?,?,1,?,?)", (f"PT-{i + 1:04d}", name, age, sex, hist, _now(), "system"))
    n = st.q("SELECT COUNT(*) AS n FROM cases", one=True)["n"]
    print(f"  Case records      : {n} saved cases in {data_dir}")


def current_user():
    u = _CURRENT_USER.get()
    return u if u else (_LOCAL_USER if not S.auth_enabled else None)


def audit(action, target="", detail=""):
    if S.store is None:
        return
    u = current_user() or {"username": "anonymous", "role": "-"}
    try:
        S.store.q("INSERT INTO audit (ts, username, role, action, target, detail) VALUES (?,?,?,?,?,?)",
                  (_now(), u["username"], u["role"], action, str(target)[:120], str(detail)[:400]))
    except Exception:
        pass


def login(username, password):
    st = S.store
    u = st.q("SELECT * FROM users WHERE username=?", (username.strip().lower(),), one=True)
    if not u or not u["active"]:
        time.sleep(0.6)
        return None
    _, h = _hash_pw(password, u["salt"])
    if not secrets.compare_digest(h, u["pw_hash"]):
        time.sleep(0.6)
        return None
    token = secrets.token_urlsafe(32)
    st.q("INSERT INTO sessions VALUES (?,?,?,?)", (token, u["username"], _now(), time.time() + SESSION_HOURS * 3600))
    st.q("UPDATE users SET last_login=? WHERE username=?", (_now(), u["username"]))
    st.q("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
    return token, {"username": u["username"], "name": u["name"], "role": u["role"]}


def session_user(token):
    if not token or S.store is None:
        return None
    s_ = S.store.q("SELECT * FROM sessions WHERE token=?", (token,), one=True)
    if not s_ or s_["expires_at"] < time.time():
        return None
    u = S.store.q("SELECT username, name, role, active FROM users WHERE username=?", (s_["username"],), one=True)
    return u if u and u["active"] else None


def _pseudo(row):
    """Researchers see patient IDs only."""
    u = current_user()
    if u and u["role"] == "researcher" and row:
        row = dict(row)
        for k in ("name", "history", "patient_name"):
            if k in row and row[k]:
                row[k] = "hidden for researchers"
    return row


def _case_summary(case):
    r = case["results"]
    top = (r.get("hybrid") or {}).get("hybrid") or r.get("classify") or {}
    cal = (r.get("hybrid") or {}).get("calibrated") or {}
    rep = r.get("report") or {}
    sg = r.get("segment") or {}
    return {"prediction": top.get("prediction"), "confidence": top.get("confidence"),
            "calibrated": cal.get("confidence"), "decision": (rep.get("decision") or {}).get("status"),
            "seg_area": sg.get("area_px") if sg.get("available") else None,
            "reliability": (rep.get("extended") or {}).get("overall_label") or (rep.get("reliability") or {}).get("label"),
            "steps": ",".join(k for k in PIPELINE if k in r)}


def persist_case(case, new=False):
    """Save the case row, its image and its results (gzip JSON) so the history survives restarts."""
    st = S.store
    if st is None:
        return
    sm = _case_summary(case)
    unc = case["results"].get("uncertainty") or {}
    qs = (unc.get("quality") or {}).get("stats") or {}
    fc = (unc.get("ood") or {}).get("feature_check") or {}
    if new:
        th = case["image"].copy()
        th.thumbnail((72, 72))
        u = current_user() or {"username": "anonymous"}
        try:
            img = case["image"].copy()
            img.thumbnail((1024, 1024))
            img.convert("RGB").save(st.case_file(case["case_id"], "jpg"), quality=92)
        except Exception:
            pass
        st.q("""INSERT OR REPLACE INTO cases (case_id, patient_id, filename, source, created_at, created_by, prediction,
                confidence, steps, status, thumb, ground_truth) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
             (case["case_id"], case.get("patient_id"), case["filename"], case.get("source"), case.get("created_at"),
              u["username"], sm["prediction"], sm["confidence"], sm["steps"], "analysed",
              _png_data_url(th, "JPEG"), case.get("ground_truth")))
        audit("case.create", case["case_id"], f"{case['filename']} -> {sm['prediction']}")
    else:
        st.q("""UPDATE cases SET prediction=?, confidence=?, calibrated=?, decision=?, seg_area=?, reliability=?, steps=?,
                q_bright=?, q_contrast=?, q_sharp=?, ood_ratio=? WHERE case_id=?""",
             (sm["prediction"], sm["confidence"], sm["calibrated"], sm["decision"], sm["seg_area"], sm["reliability"],
              sm["steps"], qs.get("brightness"), qs.get("contrast"), qs.get("sharpness"), fc.get("ratio"), case["case_id"]))
    try:
        payload = {"case_id": case["case_id"], "created_at": case.get("created_at"), "filename": case["filename"],
                   "source": case.get("source"), "ground_truth": case.get("ground_truth"),
                   "patient_id": case.get("patient_id"), "image": case["image_payload"],
                   "pixel_spacing_mm": case.get("pixel_spacing_mm"), "dicom": case.get("dicom"),
                   "results": _finite(case["results"]), "agents": _finite(case.get("agents"))}
        with gzip.open(st.case_file(case["case_id"], "json.gz"), "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:
        pass


def load_case(case_id):
    """From memory, or rebuilt from the saved files of an earlier session."""
    c = S.cases.get(case_id)
    if c is not None:
        return c
    st = S.store
    if st is None:
        raise KeyError("Case not found. Upload the MRI again.")
    p = st.case_file(case_id, "json.gz")
    if not os.path.exists(p):
        raise KeyError("Case not found. Upload the MRI again.")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    jp = st.case_file(case_id, "jpg")
    img = Image.open(jp).convert("RGB") if os.path.exists(jp) else Image.new("RGB", (224, 224))
    dw, dh = _display_size(*img.size)
    case = {"case_id": case_id, "t": time.time(), "filename": d["filename"], "source": d.get("source"),
            "ground_truth": d.get("ground_truth"), "image": img, "display_size": (dw, dh), "results": d["results"],
            "created_at": d.get("created_at"), "image_payload": d["image"], "patient_id": d.get("patient_id"),
            "pixel_spacing_mm": d.get("pixel_spacing_mm"), "dicom": d.get("dicom"), "agents": d.get("agents"),
            "restored": True}
    if "hybrid" in d["results"]:
        case["hybrid_probs"] = np.array([d["results"]["hybrid"]["hybrid"]["probabilities"][c_] / 100.0
                                         for c_ in S.class_names])
    _remember(case)
    return case


def review_case(case_id, verdict, label, note):
    u = current_user()
    if verdict not in ("agree", "correct", "uncertain"):
        raise ValueError("Choose agree, correct or uncertain.")
    if verdict == "correct" and label not in S.class_names:
        raise ValueError("Choose the corrected class.")
    row = S.store.q("SELECT * FROM cases WHERE case_id=?", (case_id,), one=True)
    if not row:
        raise KeyError("Case not found.")
    final = row["prediction"] if verdict == "agree" else (label if verdict == "correct" else None)
    S.store.q("INSERT INTO reviews (case_id, reviewer, verdict, label, note, created_at) VALUES (?,?,?,?,?,?)",
              (case_id, u["username"], verdict, final, note[:2000], _now()))
    S.store.q("""UPDATE cases SET status='reviewed', review_verdict=?, review_label=?, review_note=?, reviewed_by=?,
                 reviewed_at=? WHERE case_id=?""", (verdict, final, note[:2000], u["username"], _now(), case_id))
    audit("case.review", case_id, f"{verdict}: model {row['prediction']} -> {final}; note: {note[:120]}")
    return S.store.q("SELECT * FROM cases WHERE case_id=?", (case_id,), one=True)


def compare_cases(a, b):
    ca, cb = load_case(a), load_case(b)

    def pick(c):
        sm = _case_summary(c)
        r = c["results"]
        sg = r.get("segment") or {}
        row = S.store.q("SELECT * FROM cases WHERE case_id=?", (c["case_id"],), one=True) or {}
        return {"case_id": c["case_id"], "filename": c["filename"], "created_at": c.get("created_at"),
                "image": c["image_payload"]["src"], "overlay": sg.get("overlay") if sg.get("available") else None,
                "patient_id": c.get("patient_id"), **sm,
                "seg_percent": sg.get("area_percent_image") if sg.get("available") else None,
                "review": {k: row.get(k) for k in ("status", "review_verdict", "review_label", "reviewed_by")}}
    A, B = pick(ca), pick(cb)
    if A["created_at"] and B["created_at"] and A["created_at"] > B["created_at"]:
        A, B = B, A
    delta = {}
    if A["seg_area"] and B["seg_area"]:
        delta["seg_area_change_percent"] = (B["seg_area"] - A["seg_area"]) / A["seg_area"] * 100
    if A["confidence"] is not None and B["confidence"] is not None:
        delta["confidence_change_points"] = B["confidence"] - A["confidence"]
    delta["same_prediction"] = A["prediction"] == B["prediction"]
    notes = []
    if not delta["same_prediction"]:
        notes.append(f"The predicted class changed from {A['prediction']} to {B['prediction']}.")
    if "seg_area_change_percent" in delta:
        ch = delta["seg_area_change_percent"]
        notes.append(("The segmented region did not change in size" if abs(ch) < 0.5 else
                      f"The segmented region {'grew' if ch > 0 else 'shrank'} by {abs(ch):.0f}%")
                     + " (in pixels; both scans need the same view and resolution for this to be meaningful).")
    return {"earlier": A, "later": B, "delta": delta, "notes": notes}


def case_pdf(case):
    """A PDF case report with the images, findings, decision, checks and review status."""
    r = case["results"]
    row = (S.store.q("SELECT * FROM cases WHERE case_id=?", (case["case_id"],), one=True) if S.store else None) or {}
    pat = S.store.q("SELECT * FROM patients WHERE patient_id=?", (case.get("patient_id"),), one=True) \
        if (S.store and case.get("patient_id")) else None
    pat = _pseudo(pat) if pat else None
    rep = r.get("report") or {}
    dec = rep.get("decision") or {}
    imgs = []

    def dl(u):
        return Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB")
    imgs.append(("MRI", dl(case["image_payload"]["src"])))
    if (r.get("xai") or {}).get("overlays", {}).get("gradcam"):
        imgs.append(("Grad-CAM", dl(r["xai"]["overlays"]["gradcam"])))
    if (r.get("segment") or {}).get("available"):
        imgs.append(("U-Net segmentation", dl(r["segment"]["overlay"])))
    buf = io.BytesIO()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(buf, pagesize=A4)
        W, H = A4
        y = H - 20 * mm
        c.setFont("Helvetica-Bold", 16)
        c.drawString(18 * mm, y, "NeuroNova case report")
        c.setFont("Helvetica", 8.5)
        c.drawRightString(W - 18 * mm, y, "AI-generated research summary. Not a diagnosis.")
        y -= 8 * mm
        c.setFont("Helvetica", 9.5)
        meta = [f"Case {case['case_id']}  ·  file {case['filename']}  ·  {case.get('created_at')}",
                (f"Patient {pat['patient_id']}: {pat['name']}, {pat['age']} {pat['sex']}" + ("  (synthetic record)" if pat.get("synthetic") else ""))
                if pat else "Patient: not linked",
                f"Review: {row.get('status', 'analysed')}" + (f" by {row['reviewed_by']} — {row['review_verdict']}"
                                                               f" ({row['review_label']})" if row.get("reviewed_by") else " (not reviewed yet)")]
        for line in meta:
            c.drawString(18 * mm, y, line)
            y -= 5 * mm
        y -= 2 * mm
        x = 18 * mm
        side = (W - 36 * mm - 2 * 6 * mm) / 3
        for title, im in imgs:
            im.thumbnail((600, 600))
            c.drawImage(ImageReader(im), x, y - side, side, side, preserveAspectRatio=True, anchor="c")
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + side / 2, y - side - 4 * mm, title)
            x += side + 6 * mm
        y -= side + 10 * mm

        def para(title, lines):
            nonlocal y
            if y < 40 * mm:
                c.showPage()
                y = H - 20 * mm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(18 * mm, y, title)
            y -= 5.5 * mm
            c.setFont("Helvetica", 9)
            for ln in lines:
                words, cur = str(ln).split(), "• "
                for w in words:
                    if c.stringWidth(cur + w, "Helvetica", 9) > W - 40 * mm:
                        c.drawString(20 * mm, y, cur)
                        y -= 4.6 * mm
                        cur = "   "
                        if y < 18 * mm:
                            c.showPage()
                            y = H - 20 * mm
                            c.setFont("Helvetica", 9)
                    cur += w + " "
                c.drawString(20 * mm, y, cur)
                y -= 4.8 * mm
            y -= 2 * mm
        para("Decision (reject option)", [("RESULT WITHHELD: " + "; ".join(dec.get("reasons", []))) if dec.get("status") == "abstain"
                                          else f"Reported: calibrated confidence {dec.get('calibrated_confidence', 0):.1f}% "
                                               f"(threshold {dec.get('threshold', 0):.1f}%)"] if dec else ["Not run"])
        for key, title in (("findings", "Findings"), ("impression", "Impression"), ("measurements", "Measurements"),
                           ("recommendations", "Recommendations")):
            if rep.get(key):
                para(title, rep[key])
        ext = rep.get("extended") or {}
        checks = (rep.get("reliability") or {}).get("checks", []) + ext.get("checks", [])
        if checks:
            para(f"Reliability checks · overall {ext.get('overall_label') or (rep.get('reliability') or {}).get('label')}",
                 [("PASS  " if ch["passed"] else ("FAIL  " if ch["counted"] else "N/A   ")) + ch["name"] for ch in checks])
        if row.get("review_note"):
            para("Reviewer note", [row["review_note"]])
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(18 * mm, 12 * mm, "Research and education only. Must be reviewed by a qualified radiologist. "
                                       "Signature: ____________________")
        c.save()
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(buf) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.95, f"NeuroNova case report · {case['case_id']}", fontsize=14, weight="bold")
            for i, (title, im) in enumerate(imgs):
                ax = fig.add_axes([0.07 + i * 0.3, 0.68, 0.26, 0.24])
                ax.imshow(im)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            fig.text(0.07, 0.62, (rep.get("summary_text") or "")[:3500], fontsize=6.5, va="top", family="monospace", wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
    audit("report.pdf", case["case_id"])
    return buf.getvalue()


# ================================================================
# TOOL 12 — RESEARCH LAB
#   12A Federated learning (FedAvg) simulation across "hospitals"
#   12B Domain-shift robustness (simulated scanner changes)
#   12C Explanation faithfulness (deletion test) over many scans
#   12D Drift monitoring of the cases analysed in NeuroNova
#   Results are saved to Drive and shown in NeuroNova's research lab.
# ================================================================
RESEARCH_FILE = "neuronova_research_{}.json"
DELETION_STEPS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def deletion_curves(x, idx, maps, steps=DELETION_STEPS, seed=0):
    """Remove (blur) the most important pixels first; a faithful map makes the class probability drop fast.
    Area under the curve: lower is more faithful. A random pixel order is the baseline."""
    from PIL import ImageFilter
    base = np.asarray(x[0], dtype=np.float32)
    blurred = np.asarray(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(12)),
                         dtype=np.float32)
    n = base.shape[0] * base.shape[1]
    orders = {k: np.argsort(-np.asarray(m, dtype=np.float32).ravel(), kind="stable") for k, m in maps.items()}
    orders["random"] = np.random.default_rng(seed).permutation(n)
    keys = list(orders)
    batch = []
    for k in keys:
        for f in steps:
            m = np.zeros(n, dtype=bool)
            m[orders[k][:int(f * n)]] = True
            batch.append(np.where(m.reshape(base.shape[:2])[..., None], blurred, base))
    with S.lock:
        pr = np.asarray(S.effnet(np.stack(batch).astype(np.float32), training=False))[:, idx]
    curves, auc = {}, {}
    xs = np.asarray(steps)
    for i, k in enumerate(keys):
        c = pr[i * len(steps):(i + 1) * len(steps)]
        curves[k] = [float(v) for v in c]
        auc[k] = float(np.trapezoid(c, xs) / xs[-1]) if hasattr(np, "trapezoid") else float(np.trapz(c, xs) / xs[-1])
    best = min((k for k in keys if k != "random"), key=lambda k: auc[k])
    return {"fractions": list(steps), "curves": curves, "auc": auc, "most_faithful": best,
            "beats_random": {k: bool(auc[k] < auc["random"]) for k in keys if k != "random"}}


def _save_research(kind, res):
    S.research[kind] = res
    p = _cache_path(RESEARCH_FILE.format(kind))
    if p:
        try:
            with open(p, "w") as fh:
                json.dump(_finite(res), fh)
        except Exception:
            pass


def _load_research():
    S.research = {}
    for kind in ("federated", "domain_shift", "faithfulness"):
        p = _cache_path(RESEARCH_FILE.format(kind))
        if p and os.path.exists(p):
            try:
                with open(p) as fh:
                    S.research[kind] = json.load(fh)
            except Exception:
                pass


def _set_lab(kind, **kw):
    S.lab_status[kind] = {**S.lab_status.get(kind, {}), **kw}


def _feature_set(folder, per_class, size_cap=None, kind="train"):
    """EfficientNetV2-B0 backbone features (global average pooled) of a folder, cached to Drive."""
    rng = random.Random(7)
    items = []
    for ci, cls in enumerate(S.class_names):
        d = os.path.join(folder, cls)
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith(ALLOWED_EXT))
        if per_class:
            fs = rng.sample(fs, min(per_class, len(fs)))
        items += [(os.path.join(d, f), ci) for f in fs]
    cache = _cache_path(f"neuronova_feats_{kind}_{_fingerprint(S.effnet)}_{per_class}.npz")
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if len(z["y"]) == len(items):
            return z["X"], z["y"]
    X = []
    for s0 in range(0, len(items), 32):
        pils = [Image.open(p).convert("RGB") for p, _ in items[s0:s0 + 32]]
        with S.lock:
            X.append(features("eff", pils))
        _set_lab("federated", phase=f"extracting {kind} features {min(len(items), s0 + 32)}/{len(items)}")
    X = np.concatenate(X).astype(np.float32)
    y = np.array([c for _, c in items])
    if cache:
        np.savez_compressed(cache, X=X, y=y)
    return X, y


def _head_model(dim, n_cls, seed):
    k = _keras()
    k.utils.set_random_seed(seed)
    inp = k.Input((dim,))
    x = k.layers.Dense(256, activation="relu")(inp)
    x = k.layers.Dropout(0.3)(x)
    out = k.layers.Dense(n_cls, activation="softmax")(x)
    m = k.Model(inp, out)
    m.compile(optimizer=k.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def federated_experiment(n_clients=3, rounds=10, local_epochs=2, per_class=400, alpha=0.5, seed=0, verbose=True):
    """FedAvg (McMahan et al., 2017) simulated on one machine: each 'hospital' keeps its scans and only shares
    model weights. The classifier head is trained on frozen EfficientNetV2-B0 features, so this runs on a CPU.
    Compared with centralised training on pooled data and with each hospital training alone."""
    if S.train_dir is None or S.test_dir is None:
        raise RuntimeError("Needs the Training and Testing folders (run the dataset cell).")
    t0 = time.time()
    _set_lab("federated", state="running", phase="extracting features")
    Xtr, ytr = _feature_set(S.train_dir, per_class, kind="train")
    Xte, yte = _feature_set(S.test_dir, None, kind="test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    C_ = len(S.class_names)
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(n_clients)]
    for c in range(C_):                      # non-IID split: Dirichlet class proportions per hospital
        idx = np.where(ytr == c)[0]
        rng.shuffle(idx)
        p = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            clients[k] += part.tolist()
    clients = [np.array(sorted(c_)) for c_ in clients]
    dist = [{S.class_names[c]: int((ytr[ix] == c).sum()) for c in range(C_)} for ix in clients]
    _set_lab("federated", phase="centralised training")
    cen = _head_model(Xtr.shape[1], C_, seed)
    cen_curve = []
    for r in range(rounds):
        cen.fit(Xtr, ytr, epochs=local_epochs, batch_size=32, verbose=0)
        cen_curve.append(float(cen.evaluate(Xte, yte, verbose=0)[1]))
    local = []
    for k, ix in enumerate(clients):
        _set_lab("federated", phase=f"hospital {k + 1} training alone")
        m = _head_model(Xtr.shape[1], C_, seed + k + 1)
        if len(ix):
            m.fit(Xtr[ix], ytr[ix], epochs=rounds * local_epochs, batch_size=32, verbose=0)
        local.append({"client": k + 1, "n": int(len(ix)), "accuracy": float(m.evaluate(Xte, yte, verbose=0)[1])})
    glob = _head_model(Xtr.shape[1], C_, seed)
    fed_curve = []
    for r in range(rounds):
        _set_lab("federated", phase=f"FedAvg round {r + 1}/{rounds}")
        gw = glob.get_weights()
        new, sizes = [], []
        for k, ix in enumerate(clients):
            if not len(ix):
                continue
            m = _head_model(Xtr.shape[1], C_, seed + 100 + k)
            m.set_weights(gw)
            m.fit(Xtr[ix], ytr[ix], epochs=local_epochs, batch_size=32, verbose=0)
            new.append(m.get_weights())
            sizes.append(len(ix))
        w = np.asarray(sizes, dtype=np.float64) / sum(sizes)
        glob.set_weights([sum(wk * nw[i] for wk, nw in zip(w, new)) for i in range(len(gw))])
        fed_curve.append(float(glob.evaluate(Xte, yte, verbose=0)[1]))
        if verbose:
            print(f"  round {r + 1:>2}/{rounds}: FedAvg {fed_curve[-1] * 100:5.1f}%  ·  centralised {cen_curve[r] * 100:5.1f}%")
    res = {"generated_at": _now(), "seconds": round(time.time() - t0, 1), "clients": n_clients, "rounds": rounds,
           "local_epochs": local_epochs, "alpha": alpha, "per_class": per_class, "train_images": int(len(ytr)),
           "test_images": int(len(yte)), "client_distribution": dist, "fedavg_curve": fed_curve,
           "centralised_curve": cen_curve, "local_only": local,
           "summary": {"fedavg": fed_curve[-1], "centralised": cen_curve[-1],
                       "local_mean": float(np.mean([l_["accuracy"] for l_ in local])),
                       "local_best": float(max(l_["accuracy"] for l_ in local))},
           "note": ("Simulation on one machine: the training scans are split between hospitals with different "
                    "class mixes (Dirichlet alpha {}), only model weights are averaged, and every model is a "
                    "classifier head on frozen EfficientNetV2-B0 features.".format(alpha))}
    _save_research("federated", res)
    _set_lab("federated", state="done", phase="")
    return res


SHIFTS = [
    ("Original", lambda im: im),
    ("Darker (brightness -35%)", lambda im: ImageEnhance.Brightness(im).enhance(0.65)),
    ("Low contrast (x0.55)", lambda im: ImageEnhance.Contrast(im).enhance(0.55)),
    ("Gamma 1.8 (different scanner curve)", lambda im: Image.fromarray(
        (255 * (np.asarray(im, dtype=np.float32) / 255) ** 1.8).astype(np.uint8))),
    ("Gaussian noise (sigma 14)", lambda im: Image.fromarray(np.clip(
        np.asarray(im, dtype=np.float32) + np.random.default_rng(1).normal(0, 14, np.asarray(im).shape), 0, 255).astype(np.uint8))),
    ("Blur (radius 2.5)", lambda im: im.filter(ImageFilter.GaussianBlur(2.5))),
    ("Low resolution (1/4 size)", lambda im: im.resize((max(8, im.size[0] // 4), max(8, im.size[1] // 4))).resize(im.size)),
    ("Heavy JPEG compression (q=12)", lambda im: _jpeg(im, 12)),
    ("Bias field (uneven coil sensitivity)", lambda im: _bias(im)),
]


def _jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _bias(im):
    a = np.asarray(im, dtype=np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    field = 0.6 + 0.8 * (xx / max(1, w - 1)) * 0.6 + 0.4 * (yy / max(1, h - 1)) * 0.5
    return Image.fromarray(np.clip(a * field[..., None], 0, 255).astype(np.uint8))


def domain_shift_test(n_per_class=25, seed=0, verbose=True):
    """How accuracy and the safety checks behave when the scans look as if they came from another scanner."""
    if S.test_dir is None:
        raise RuntimeError("Needs the Testing folder (run the dataset cell).")
    t0 = time.time()
    _set_lab("domain_shift", state="running", phase="loading Testing images")
    rng = random.Random(seed)
    items = []
    for ci, cls in enumerate(S.class_names):
        d = os.path.join(S.test_dir, cls)
        if os.path.isdir(d):
            fs = sorted(f for f in os.listdir(d) if f.lower().endswith(ALLOWED_EXT))
            items += [(os.path.join(d, f), ci) for f in rng.sample(fs, min(n_per_class, len(fs)))]
    pils = [Image.open(p).convert("RGB") for p, _ in items]
    y = np.array([c for _, c in items])
    rs = reliability_settings()
    T = float(rs["temperatures"].get("hybrid" if S.inception is not None else "efficientnet", 1.0))
    rows = []
    for name, fn in SHIFTS:
        _set_lab("domain_shift", phase=name)
        shifted = [fn(im).convert("RGB") for im in pils]
        with S.lock:
            hyb, _, _ = predict_hybrid(shifted)
        hc = temper(hyb, T)
        pred = hc.argmax(1)
        conf = hc.max(1)
        flagged = []
        ratios = []
        for im in shifted:
            q = validate_input(im)
            ratio = None
            if S.index is not None:
                with S.lock:
                    f = features("eff", [im])[0]
                d_, p99, _ = _ood_distance(f)
                ratio = d_ / p99 if p99 > 0 else None
            ratios.append(ratio)
            flagged.append(q["verdict"] == "fail" or (ratio is not None and ratio > 1.25))
        withheld = [(fl or c < rs["threshold"]) for fl, c in zip(flagged, conf)]
        acc = float((pred == y).mean())
        kept = ~np.array(withheld)
        rows.append({"shift": name, "accuracy": acc, "mean_confidence": float(conf.mean()),
                     "flagged_rate": float(np.mean(flagged)), "withheld_rate": float(np.mean(withheld)),
                     "accuracy_of_reported": float((pred[kept] == y[kept]).mean()) if kept.any() else None,
                     "mean_ood_ratio": float(np.mean([r for r in ratios if r is not None])) if any(r is not None for r in ratios) else None})
        if verbose:
            r = rows[-1]
            print(f"  {name:<38} accuracy {r['accuracy'] * 100:5.1f}%  confidence {r['mean_confidence'] * 100:5.1f}%"
                  f"  flagged {r['flagged_rate'] * 100:5.1f}%  withheld {r['withheld_rate'] * 100:5.1f}%")
    base = rows[0]["accuracy"]
    res = {"generated_at": _now(), "seconds": round(time.time() - t0, 1), "images": int(len(y)), "rows": rows,
           "worst": min(rows[1:], key=lambda r: r["accuracy"])["shift"],
           "note": ("Each shift imitates a different scanner or protocol. A good system either keeps its accuracy "
                    "or flags the scans as unusual so that the reject option can withhold them; a confident wrong "
                    f"answer without a flag is the dangerous case. Baseline accuracy {base * 100:.1f}%.")}
    _save_research("domain_shift", res)
    _set_lab("domain_shift", state="done", phase="")
    return res


def faithfulness_benchmark(n=40, seed=0, verbose=True):
    """Deletion test over many Testing scans: which explanation method is most faithful to the model?"""
    if S.test_dir is None:
        raise RuntimeError("Needs the Testing folder (run the dataset cell).")
    t0 = time.time()
    _set_lab("faithfulness", state="running", phase="explaining")
    rng = random.Random(seed)
    items = []
    for cls in S.class_names:
        d = os.path.join(S.test_dir, cls)
        if os.path.isdir(d) and cls.lower() not in ("notumor", "no_tumor"):
            fs = sorted(f for f in os.listdir(d) if f.lower().endswith(ALLOWED_EXT))
            items += [os.path.join(d, f) for f in rng.sample(fs, min(max(1, n // 3), len(fs)))]
    items = items[:n]
    aucs, curves = {}, {}
    _ensure_grad_model()
    for i, p in enumerate(items):
        img = Image.open(p).convert("RGB").resize((224, 224))
        x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)
        with S.lock:
            idx = int(np.argmax(np.asarray(S.effnet(x, training=False))[0]))
            heat, mx = _gradcam_raw(x, idx)
            if mx <= 0:
                heat, mx = _gradcam_raw(x, idx, use_logit=True)
            maps = {"gradcam": _resize_map(_norm_map(heat if heat is not None else np.zeros((7, 7)))),
                    "gradcam_pp": gradcam_plus_plus(x, idx), "integrated_gradients": integrated_gradients(x, idx)}
        maps["consensus"] = _norm_map(np.mean(list(maps.values()), 0))
        d = deletion_curves(x, idx, maps, seed=i)
        for k, v in d["auc"].items():
            aucs.setdefault(k, []).append(v)
            curves.setdefault(k, []).append(d["curves"][k])
        _set_lab("faithfulness", phase=f"{i + 1}/{len(items)} scans")
        if verbose:
            print(f"  explained {i + 1}/{len(items)}", end="\r", flush=True)
    names = {"gradcam": "Grad-CAM", "gradcam_pp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients",
             "consensus": "Consensus", "random": "Random order (baseline)"}
    summary = {k: {"label": names[k], "auc": float(np.mean(v)), "auc_std": float(np.std(v)),
                   "curve": [float(c) for c in np.mean(curves[k], 0)],
                   "beats_random_share": float(np.mean([a < r for a, r in zip(v, aucs["random"])])) if k != "random" else None}
               for k, v in aucs.items()}
    res = {"generated_at": _now(), "seconds": round(time.time() - t0, 1), "n": len(items),
           "fractions": list(DELETION_STEPS), "summary": summary,
           "best": min((k for k in summary if k != "random"), key=lambda k: summary[k]["auc"]),
           "note": "Lower area under the deletion curve means the method found the pixels the model really relies on."}
    _save_research("faithfulness", res)
    _set_lab("faithfulness", state="done", phase="")
    if verbose:
        print(" " * 40)
    return res


def _psi(ref, cur, bins=10):
    """Population stability index with quantile bins of the reference; fewer bins and add-half smoothing
    for small samples, so a handful of cases does not produce huge values from empty bins."""
    ref, cur = np.asarray(ref, dtype=np.float64), np.asarray(cur, dtype=np.float64)
    bins = int(min(bins, max(3, len(cur) // 4)))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    k = len(edges) - 1
    r = (np.histogram(ref, edges)[0] + 0.5) / (len(ref) + 0.5 * k)
    c = (np.histogram(cur, edges)[0] + 0.5) / (len(cur) + 0.5 * k)
    raw = float(np.sum((c - r) * np.log(c / r)))
    # PSI is biased upwards for small samples (about (k - 1)(1/n + 1/m) when nothing changed); remove that part
    return max(0.0, raw - (k - 1) * (1.0 / max(1, len(cur)) + 1.0 / max(1, len(ref))))


def _drift_level(psi, p):
    if psi > 0.25 and p < 0.05:
        return "drift"
    if psi > 0.1 or p < 0.05:
        return "moderate"
    return "stable"


def monitoring_report(window=50):
    """Population stability index (PSI) of recent cases against the training scans, and trends over time."""
    if S.store is None:
        return {"available": False, "message": "No case database in this session."}
    rows = S.store.q("SELECT * FROM cases WHERE prediction IS NOT NULL ORDER BY created_at DESC LIMIT ?", (int(window),))
    rows = rows[::-1]
    out = {"available": True, "n": len(rows), "window": window, "features": [], "alerts": [],
           "series": [{"t": r["created_at"], "case_id": r["case_id"], "prediction": r["prediction"],
                       "confidence": r["confidence"], "calibrated": r["calibrated"], "decision": r["decision"],
                       "ood_ratio": r.get("ood_ratio"), "brightness": r.get("q_bright")} for r in rows]}
    if len(rows) < 5:
        out["message"] = "Analyse at least 5 scans to see drift statistics."
        return out
    ix = S.index or {}
    refs = [("q_bright", "Brightness", ix.get("q_bright")), ("q_contrast", "Contrast", ix.get("q_contrast")),
            ("q_sharp", "Sharpness", ix.get("q_sharp")),
            ("ood_ratio", "Distance from training scans",
             (np.asarray(ix["d_train"]) / np.percentile(ix["d_train"], 99)) if ix.get("d_train") is not None else None)]
    for key, label, ref in refs:
        cur = [r.get(key) for r in rows if r.get(key) is not None]
        if ref is None or len(cur) < 5:
            continue
        psi = _psi(ref, cur)
        try:
            from scipy.stats import ks_2samp
            pval = float(ks_2samp(np.asarray(ref, dtype=np.float64), np.asarray(cur, dtype=np.float64)).pvalue)
        except Exception:
            pval = 1.0
        level = _drift_level(psi, pval)
        out["features"].append({"key": key, "label": label, "psi": psi, "ks_p": pval, "level": level,
                                "reference_median": float(np.median(ref)), "recent_median": float(np.median(cur))})
        if level != "stable":
            out["alerts"].append(f"{label}: PSI {psi:.2f}, KS test p = {pval:.3f} ({level}) — recent median "
                                 f"{np.median(cur):.1f} vs training {np.median(ref):.1f}.")
    counts = (S.eval_result or {}).get("class_counts")
    if not counts and S.test_dir:
        counts = {c: len(os.listdir(os.path.join(S.test_dir, c))) for c in S.class_names
                  if os.path.isdir(os.path.join(S.test_dir, c))}
    cur = {c: sum(1 for r in rows if r["prediction"] == c) for c in S.class_names}
    if counts:
        tot_r, tot_c = sum(counts.values()), max(1, len(rows))
        k_ = len(S.class_names)
        r_ = (np.array([counts.get(c, 0) for c in S.class_names]) + 0.5) / (tot_r + 0.5 * k_)
        c_ = (np.array([cur[c] for c in S.class_names]) + 0.5) / (tot_c + 0.5 * k_)
        psi = max(0.0, float(np.sum((c_ - r_) * np.log(c_ / r_))) - (k_ - 1) * (1 / tot_c + 1 / tot_r))
        try:
            from scipy.stats import chisquare
            exp = np.array([counts.get(c, 0) / tot_r for c in S.class_names]) * tot_c
            pval = float(chisquare([cur[c] for c in S.class_names], np.maximum(exp, 1e-9)).pvalue)
        except Exception:
            pval = 1.0
        level = _drift_level(psi, pval)
        out["prediction_mix"] = {"reference": {c: counts.get(c, 0) / tot_r for c in S.class_names},
                                 "recent": {c: cur[c] / tot_c for c in S.class_names}, "psi": psi, "chi2_p": pval,
                                 "level": level}
        if level == "drift":
            out["alerts"].append(f"Prediction mix changed (PSI {psi:.2f}, chi-square p = {pval:.3f}): recent scans "
                                 "are predicted differently from the Testing folder's class balance.")
    wr = [r["decision"] == "abstain" for r in rows if r["decision"]]
    out["withheld_rate"] = float(np.mean(wr)) if wr else None
    out["mean_calibrated"] = float(np.mean([r["calibrated"] for r in rows if r["calibrated"] is not None])) \
        if any(r["calibrated"] is not None for r in rows) else None
    half = len(rows) // 2
    if half >= 5 and wr:
        a = np.mean([r["decision"] == "abstain" for r in rows[:half] if r["decision"]] or [0])
        b = np.mean([r["decision"] == "abstain" for r in rows[half:] if r["decision"]] or [0])
        if b - a > 0.2:
            out["alerts"].append(f"More results are being withheld lately ({a * 100:.0f}% → {b * 100:.0f}%).")
    return out


def show_research(kind=None):
    import matplotlib.pyplot as plt
    R = S.research
    if (kind in (None, "federated")) and R.get("federated"):
        f = R["federated"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2))
        xs = np.arange(1, f["rounds"] + 1)
        a1.plot(xs, np.array(f["fedavg_curve"]) * 100, "o-", label="FedAvg (weights only)")
        a1.plot(xs, np.array(f["centralised_curve"]) * 100, "s--", label="Centralised (all data pooled)")
        for l_ in f["local_only"]:
            a1.axhline(l_["accuracy"] * 100, ls=":", lw=1, color="grey")
            a1.text(f["rounds"], l_["accuracy"] * 100, f" hospital {l_['client']} alone", fontsize=7, va="center")
        a1.set_xlabel("Round")
        a1.set_ylabel("Testing accuracy (%)")
        a1.set_title("12A — Federated learning (FedAvg)")
        a1.legend(fontsize=8)
        bottom = np.zeros(len(f["client_distribution"]))
        for c in S.class_names:
            v = np.array([d_.get(c, 0) for d_ in f["client_distribution"]])
            a2.bar([f"Hospital {i + 1}" for i in range(len(v))], v, bottom=bottom, label=c)
            bottom += v
        a2.set_title(f"Non-IID data: class mix per hospital (Dirichlet alpha {f['alpha']})")
        a2.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        s_ = f["summary"]
        print(f"FedAvg {s_['fedavg'] * 100:.1f}% · centralised {s_['centralised'] * 100:.1f}% · "
              f"hospitals alone {s_['local_mean'] * 100:.1f}% on average (best {s_['local_best'] * 100:.1f}%)")
        print(" ", f["note"])
    if (kind in (None, "domain_shift")) and R.get("domain_shift"):
        d = R["domain_shift"]
        rows = d["rows"]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        yy = np.arange(len(rows))
        ax.barh(yy - 0.2, [r["accuracy"] * 100 for r in rows], 0.4, label="accuracy")
        ax.barh(yy + 0.2, [r["flagged_rate"] * 100 for r in rows], 0.4, label="flagged by the safety checks")
        ax.set_yticks(yy, [r["shift"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("%")
        ax.set_title(f"12B — Domain shift ({d['images']} Testing scans per condition)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        print(" ", d["note"])
    if (kind in (None, "faithfulness")) and R.get("faithfulness"):
        fa = R["faithfulness"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for k, v in fa["summary"].items():
            ax.plot(np.array(fa["fractions"]) * 100, v["curve"], "--" if k == "random" else "o-",
                    label=f"{v['label']} (AUC {v['auc']:.3f})")
        ax.set_xlabel("Share of pixels removed, most important first (%)")
        ax.set_ylabel("Probability of the predicted class")
        ax.set_title(f"12C — Deletion test over {fa['n']} scans (lower curve = more faithful)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        print(f"  Most faithful: {fa['summary'][fa['best']]['label']}. {fa['note']}")


# ---------------- notebook helpers (analyse a file, show results with matplotlib) ----------------

def new_case_from_image(img, filename="image.png", ground_truth=None, source="notebook"):
    case_id = "CASE-" + datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8].upper()
    dw, dh = _display_size(*img.size)
    disp = img.resize((dw, dh), Image.Resampling.LANCZOS) if (dw, dh) != img.size else img
    case = {"case_id": case_id, "t": time.time(), "filename": filename, "source": source,
            "ground_truth": ground_truth, "image": img, "display_size": (dw, dh), "results": {},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "image_payload": {"src": _png_data_url(disp, "JPEG"), "width": img.size[0], "height": img.size[1],
                              "display_width": dw, "display_height": dh}}
    return case


def analyze_file(path, verbose=True):
    """Run every NeuroNova step on one image file; the case also appears in the web UI's history."""
    img = Image.open(path).convert("RGB")
    case = new_case_from_image(img, os.path.basename(path))
    with S.lock:
        case["results"]["classify"] = stage_classify(case)
    _remember(case)
    persist_case(case, new=True)
    for key in PIPELINE:
        if key == "classify":
            continue
        t = time.perf_counter()
        fn = STAGES[key]
        if getattr(fn, "self_locking", False):
            case["results"][key] = fn(case)
        else:
            with S.lock:
                case["results"][key] = fn(case)
        if verbose:
            print(f"  {key:<12} done in {time.perf_counter() - t:5.1f} s")
    persist_case(case)
    return case


def _show_data_url(ax, url, title):
    raw = base64.b64decode(url.split(",", 1)[1])
    ax.imshow(Image.open(io.BytesIO(raw)))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def show_uncertainty(case):
    import matplotlib.pyplot as plt
    u = case["results"]["uncertainty"]
    h = u["hybrid"]
    print(f"TOOL 8A — Monte Carlo dropout ({u['samples']} passes, {u['method']})")
    print(f"  MC prediction      : {h['prediction']}  (deterministic: {u['deterministic_prediction']})")
    print(f"  Epistemic (model)  : {h['epistemic']:.4f}  -> {u['epistemic_level']} uncertainty")
    print(f"  Aleatoric (data)   : {h['aleatoric']:.4f}")
    print(f"  Predictive entropy : {h['predictive_entropy']:.4f}")
    o = u["ood"]
    print(f"\nOut-of-distribution check: {o['label']}")
    for r in o["reasons"]:
        print("  -", r)
    if o["feature_check"]:
        fc = o["feature_check"]
        print(f"  Feature distance {fc['distance']:.2f} vs 99th percentile of training scans {fc['p99']:.2f}"
              f" (further than {fc['percentile']:.1f}% of them)")
    q = u.get("quality")
    if q:
        print(f"\nInput quality: {q['label']} (compared with {q['reference']})")
        for c in q["checks"]:
            print(f"  [{c['status'].upper():4}] {c['name']:<11} {c['value']:<42} {c['detail']}")
    rep = case["results"].get("report") or {}
    dec = rep.get("decision")
    if dec:
        print(f"\nDecision (reject option): {'RESULT WITHHELD' if dec['status'] == 'abstain' else 'RESULT REPORTED'}"
              f" — calibrated confidence {dec['calibrated_confidence']:.1f}%, threshold {dec['threshold']:.1f}%")
        for r_ in dec["reasons"]:
            print("  -", r_)
    cls = S.class_names
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(cls, [h["mean"][c] for c in cls], yerr=[h["std"][c] for c in cls], capsize=6, color="#4C78A8")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Hybrid prediction: mean ± std over Monte Carlo dropout passes")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.show()


def show_xai_lab(case):
    import matplotlib.pyplot as plt
    x = case["results"]["xai"]
    titles = {"gradcam": "Grad-CAM", "gradcam_pp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients",
              "lime": "LIME", "consensus": "Consensus"}
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, k in zip(axes, x["methods"]):
        _show_data_url(ax, x["overlays"][k], titles[k])
    plt.suptitle(f"TOOL 8B — Explainability lab: {x['strength']} (mean overlap {x['mean_pairwise_iou']:.2f})")
    plt.tight_layout()
    plt.show()
    names = x["methods"][:4]
    print("Top-20% overlap (IoU) between methods:")
    print(" " * 22 + "".join(f"{titles[b][:12]:>14}" for b in names))
    for a in names:
        print(f"{titles[a]:<22}" + "".join(f"{x['iou'][a + '|' + b]:>14.2f}" for b in names))
    fa = x.get("faithfulness") or {}
    if fa.get("curves"):
        titles["random"] = "Random order (baseline)"
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for k, c in fa["curves"].items():
            ax.plot(np.array(fa["fractions"]) * 100, c, "--" if k == "random" else "o-",
                    label=f"{titles.get(k, k)} (AUC {fa['auc'][k]:.3f})")
        ax.set_xlabel("Share of pixels removed, most important first (%)")
        ax.set_ylabel("Probability of the predicted class")
        ax.set_title("Deletion test: lower curve = more faithful explanation")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        print(f"Most faithful for this scan: {titles.get(fa['most_faithful'], fa['most_faithful'])}")


def show_similar(case):
    import matplotlib.pyplot as plt
    s = case["results"]["similar"]
    if not s.get("available"):
        print("TOOL 8C —", s.get("message"))
        return
    fig, axes = plt.subplots(1, len(s["neighbours"]), figsize=(3.2 * len(s["neighbours"]), 3.6))
    for ax, n in zip(np.atleast_1d(axes), s["neighbours"]):
        _show_data_url(ax, n["thumb"], f"{n['label']}\n{n['similarity']:.1f}% similar")
    plt.suptitle(f"TOOL 8C — Most similar training scans ({int(s['agreement'] * 100)}% of the 10 nearest are "
                 f"{s['prediction']})")
    plt.tight_layout()
    plt.show()
    if s["duplicate"]:
        print("  Warning: an almost identical image is in the training set.")
    if s.get("warning"):
        print("  Note:", s["warning"])


def show_evaluation(ev=None):
    import matplotlib.pyplot as plt
    ev = ev or S.eval_result
    cls = ev["classes"]
    print(f"TOOL 8D — Evaluation on {ev['n_images']} Testing images ({ev['seconds']} s)")
    print(f"  {ev['note']}\n")
    print(f"{'Model':<24}{'Accuracy':>10}{'Macro F1':>10}{'Macro AUC':>11}{'Kappa':>8}{'MCC':>8}{'ECE':>8}")
    for k in ev["order"]:
        m = ev["models"][k]
        print(f"{m['label']:<24}{m['accuracy'] * 100:>9.2f}%{m['macro_f1']:>10.3f}{m['macro_auc']:>11.3f}"
              f"{m['kappa']:>8.3f}{m['mcc']:>8.3f}{m['ece']:>8.3f}")
    for k in ev["order"]:
        m = ev["models"][k]
        print(f"\n{m['label']} — per-class report")
        print(f"  {'Class':<12}{'Precision':>10}{'Recall':>9}{'F1':>8}{'AUC':>8}{'Support':>9}")
        for r in m["per_class"]:
            print(f"  {r['class']:<12}{r['precision']:>10.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}"
                  f"{r['auc']:>8.3f}{r['support']:>9d}")
    keys = ev["order"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.2))
    for ax, k in zip(np.atleast_1d(axes), keys):
        cm = np.array(ev["models"][k]["confusion"])
        ax.imshow(cm, cmap="Blues")
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks(range(len(cls)), cls, rotation=30, ha="right")
        ax.set_yticks(range(len(cls)), cls)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{ev['models'][k]['label']}\naccuracy {ev['models'][k]['accuracy'] * 100:.2f}%")
    plt.suptitle("Confusion matrices")
    plt.tight_layout()
    plt.show()
    key = "hybrid" if "hybrid" in ev["models"] else keys[0]
    m = ev["models"][key]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for c in cls:
        pts = np.array(m["roc"][c])
        auc = next(r["auc"] for r in m["per_class"] if r["class"] == c)
        a1.plot(pts[:, 0], pts[:, 1], label=f"{c} (AUC {auc:.3f})")
    a1.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a1.set_title(f"ROC curves — {m['label']}")
    a1.legend(fontsize=8)
    cal = m["calibration"]
    a2.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    a2.plot([b["confidence"] for b in cal], [b["accuracy"] for b in cal], "o-", label=m["label"])
    a2.set_xlabel("Mean confidence")
    a2.set_ylabel("Accuracy")
    a2.set_title(f"Reliability diagram (ECE {m['ece']:.3f})")
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    if ev["mistakes"]:
        n = len(ev["mistakes"])
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.9))
        for ax, mk in zip(np.atleast_1d(axes), ev["mistakes"]):
            if mk["thumb"]:
                _show_data_url(ax, mk["thumb"], f"true {mk['true']}\npred {mk['predicted']} {mk['confidence']:.0f}%")
        plt.suptitle(f"Most confident mistakes — {ev['mistakes_model']}")
        plt.tight_layout()
        plt.show()
    show_reliability(ev)


def show_reliability(ev=None):
    """Precision-recall curves, temperature scaling, the reject option and the speed benchmark."""
    import matplotlib.pyplot as plt
    ev = ev or S.eval_result
    cls = ev["classes"]
    key = "hybrid" if "hybrid" in ev["models"] else ev["order"][0]
    m = ev["models"][key]
    cal = ev.get("calibration")
    fig, axes = plt.subplots(1, 3 if cal else 1, figsize=(18 if cal else 6, 4.6))
    axes = np.atleast_1d(axes)
    a0 = axes[0]
    for c in cls:
        pts = np.array(m["pr"][c])
        ap = next(r["ap"] for r in m["per_class"] if r["class"] == c)
        a0.plot(pts[:, 0], pts[:, 1], label=f"{c} (AP {ap:.3f})")
    a0.set_xlabel("Recall")
    a0.set_ylabel("Precision")
    a0.set_ylim(0, 1.02)
    a0.set_title(f"Precision-recall — {m['label']}")
    a0.legend(fontsize=8)
    if cal:
        d = cal["decision"]
        cm_ = cal["models"][d["model"]]
        a1 = axes[1]
        for bins, lab, st in ((cm_["reliability_before"], f"before (ECE {cm_['before']['ece']:.3f})", "o-"),
                              (cm_["reliability_after"], f"after T={cm_['temperature']:.2f} (ECE {cm_['after']['ece']:.3f})", "s-")):
            a1.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], st, label=lab)
        a1.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
        a1.set_xlabel("Mean confidence")
        a1.set_ylabel("Accuracy")
        a1.set_title(f"Temperature scaling — {d['label']}")
        a1.legend(fontsize=8)
        a2 = axes[2]
        rc = np.array(d["risk_coverage"])
        a2.plot(rc[:, 0], rc[:, 1], color="#4C78A8")
        a2.axhline(d["target_accuracy"], ls="--", color="grey", lw=1, label=f"target {d['target_accuracy'] * 100:.1f}%")
        a2.axvline(d["report"]["coverage"], ls=":", color="#E45756", lw=1.5,
                   label=f"chosen threshold: {d['report']['coverage'] * 100:.0f}% of cases reported")
        a2.set_xlabel("Coverage (share of cases reported)")
        a2.set_ylabel("Accuracy of reported cases")
        a2.set_title("Reject option (held-out half of Testing)")
        a2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    if cal:
        d, r = cal["decision"], cal["decision"]["report"]
        print(f"TEMPERATURE SCALING (fitted on {cal['split']['calibration']} Testing images, "
              f"reported on the other {cal['split']['report']})")
        print(f"  {'Model':<24}{'T':>7}{'ECE before':>12}{'ECE after':>11}{'NLL before':>12}{'NLL after':>11}")
        for k in ev["order"]:
            c = cal["models"][k]
            print(f"  {c['label']:<24}{c['temperature']:>7.3f}{c['before']['ece']:>12.3f}{c['after']['ece']:>11.3f}"
                  f"{c['before']['log_loss']:>12.3f}{c['after']['log_loss']:>11.3f}")
        print(f"\nREJECT OPTION ({d['label']}, calibrated)")
        print(f"  Target accuracy of reported cases : {d['target_accuracy'] * 100:.1f}% (half the error rate)"
              + ("" if d["target_reached"] else "  [not reachable; threshold set at 50% coverage]"))
        print(f"  Threshold on calibrated confidence: {d['threshold'] * 100:.1f}%")
        print(f"  Held-out half: {r['coverage'] * 100:.1f}% of cases reported with accuracy "
              f"{(r['selective_accuracy'] or 0) * 100:.1f}% (all cases: {r['full_accuracy'] * 100:.1f}%)")
        if r["rejected"]:
            print(f"  {r['rejected']} cases withheld; their accuracy would have been {(r['rejected_accuracy'] or 0) * 100:.1f}%")
    b = ev.get("benchmark") or {}
    if b and "error" not in b:
        print(f"\nSPEED AND SIZE (this Colab runtime)")
        print(f"  {'Model':<24}{'Parameters':>13}{'Weights MB':>12}{'File MB':>9}{'ms / image':>12}{'images / s':>12}")
        for k, v in b.items():
            fmb = f"{v['file_mb']:.1f}" if v.get("file_mb") else "-"
            print(f"  {v['label']:<24}{v['parameters']:>13,}{v['weights_mb']:>12.1f}{fmb:>9}"
                  f"{v['latency_ms']:>12.1f}{v['throughput']:>12.1f}")


stage_uncertainty.self_locking = True
stage_similar.self_locking = True
stage_xai.self_locking = True
stage_segment.self_locking = True


STAGES = {
    "gradcam": stage_gradcam,
    "hybrid": stage_hybrid,
    "stability": stage_stability,
    "occlusion": stage_occlusion,
    "uncertainty": stage_uncertainty,
    "xai": stage_xai,
    "similar": stage_similar,
    "segment": stage_segment,
    "report": stage_report,
}
PIPELINE = ["classify", "gradcam", "hybrid", "stability", "occlusion", "uncertainty", "xai", "similar", "segment",
            "report"]


# ----------------------------------------------------------------
# Assistant (Tool 6 prompt + case context; Tool 4 / offline fallback)
# ----------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """
You are an educational Brain Tumour and Explainable AI Assistant powered by Gemini,
built into the NeuroNova web interface of a student research project.
You support students, patients, caregivers, and researchers.
Answer in clear, structured language and explain unfamiliar terms.

The project: EfficientNetV2-B0 (Tool 1) classifies a brain MRI into glioma,
meningioma, notumor or pituitary. Grad-CAM (Tool 2) shows which image regions
drove the prediction. A hybrid soft vote with InceptionV3 (Tool 3), a stability
test over 10 image variations (Tool 3B) and occlusion sensitivity compared with
Grad-CAM (Tool 3C) feed an AI-generated summary with a reliability grade (Tool 3D).
Tool 8 adds Monte Carlo dropout uncertainty and an out-of-distribution check (8A),
Grad-CAM++, Integrated Gradients, LIME and a consensus map (8B), the most similar
training scans (8C), test-set evaluation metrics (8D) and batch analysis (8E).

MEDICAL SAFETY
- This is educational information, not a diagnosis or a substitute for a
  qualified healthcare professional.
- Never invent the contents of a user's report, MRI, or project results.
- If case data are missing, say they are not currently available.
- Do not treat an AI class label, confidence score, or Grad-CAM heatmap as
  proof of tumour presence, exact grade, or clinical outcome.
- Never estimate an individual patient's remaining days, months, or years.
  If asked, respond compassionately, explain that these data cannot determine
  an individual outcome, list the factors doctors use (confirmed tumour type,
  grade, molecular markers, age, location, general health, treatment response)
  and recommend the treating neuro-oncology team.
- Published survival figures are group-level statistics. Do not fabricate
  numbers or citations.
- Do not prescribe medication, doses, or an individual treatment plan.
- For a sudden seizure, loss of consciousness, sudden weakness, new speech
  difficulty, or sudden severe headache, advise immediate emergency care.
- Be compassionate; avoid frightening, absolute, or falsely reassuring language.
- Do not claim you looked at the image: you only receive the numbers below.
"""


def _case_context(case):
    if not case:
        return "No MRI case has been analysed yet in this session."
    r = case["results"]
    ctx = {"filename": case.get("filename")}
    if case.get("ground_truth"):
        ctx["dataset_folder_label"] = case["ground_truth"]
    if "classify" in r:
        c = r["classify"]
        ctx["tool1_efficientnet"] = {k: c[k] for k in ("prediction", "confidence", "confidence_level",
                                                       "probabilities", "quality")}
    if "gradcam" in r:
        g = r["gradcam"]
        ctx["tool2_gradcam"] = {k: g[k] for k in ("layer", "activation_percent", "mean_activation",
                                                  "max_activation", "intensity", "bbox", "original_size")}
    if "hybrid" in r:
        h = dict(r["hybrid"])
        h.pop("ms", None)
        ctx["tool3_hybrid"] = h
    if "stability" in r:
        s = dict(r["stability"])
        s.pop("ms", None)
        ctx["tool3b_stability"] = s
    if "occlusion" in r:
        o = {k: v for k, v in r["occlusion"].items()
             if k not in ("occlusion_overlay", "gradcam_overlay", "overlap_mask", "ms")}
        ctx["tool3c_occlusion"] = o
    if "report" in r:
        ctx["tool3d_report"] = {k: r["report"][k] for k in ("findings", "impression", "recommendations",
                                                            "reliability")}
        if r["report"].get("extended"):
            ctx["tool8_extended_checks"] = r["report"]["extended"]
    if "uncertainty" in r:
        u = r["uncertainty"]
        ctx["tool8a_uncertainty"] = {"mc_passes": u["samples"], "hybrid_mc": u["hybrid"],
                                     "epistemic_level": u["epistemic_level"],
                                     "out_of_distribution": {k: u["ood"][k] for k in
                                                             ("label", "reasons", "image", "feature_check")}}
    if "xai" in r:
        x = r["xai"]
        ctx["tool8b_explainability"] = {"methods": x["methods"], "mean_pairwise_iou": x["mean_pairwise_iou"],
                                        "strength": x["strength"], "iou": x["iou"],
                                        "consensus_bbox_224px": x["consensus_bbox"]}
    if "similar" in r and r["similar"].get("available"):
        sm = r["similar"]
        ctx["tool8c_similar_cases"] = {"votes_of_10_nearest": sm["votes"], "agreement": sm["agreement"],
                                       "nearest": [{"label": n["label"], "similarity": n["similarity"]}
                                                   for n in sm["neighbours"]],
                                       "near_duplicate_in_training": sm["duplicate"]}
    if S.eval_result:
        ctx["tool8d_test_set_metrics"] = {
            v["label"]: {k: v[k] for k in ("accuracy", "macro_f1", "macro_auc", "kappa", "ece")}
            for v in S.eval_result["models"].values()}
        ctx["tool8d_note"] = S.eval_result.get("note")
    return json.dumps(ctx, indent=1, default=str)


def _history_text(history, max_messages=10, max_chars=1500):
    lines = []
    for m in (history or [])[-max_messages:]:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            role = m.get("role") if m.get("role") in ("user", "assistant") else "user"
            lines.append(f"{role.upper()}: {m['content'][:max_chars]}")
    return "\n".join(lines) or "No previous conversation."


def chat_answer(message, history, case, mode, patient):
    mode_line = (
        "Answer for a patient or family member: plain words, short paragraphs, explain every technical term."
        if mode == "patient" else
        "Answer for a doctor, researcher or examiner: precise terminology, include the numbers and method details."
    )
    patient_ctx = ""
    if patient:
        safe = {k: v for k, v in patient.items() if k in ("age", "sex", "history") and v}
        if safe:
            patient_ctx = "\nPATIENT CONTEXT SUPPLIED BY THE USER:\n" + json.dumps(safe)

    if S.gemini_client is not None and S.gemini_types is not None:
        prompt = f"""{AGENT_SYSTEM_PROMPT}

AUDIENCE: {mode_line}

CURRENT CASE RESULTS FROM THE PROJECT TOOLS (numbers only):
{_case_context(case)}
{patient_ctx}

RECENT CONVERSATION:
{_history_text(history)}

CURRENT USER QUESTION:
{message}

Answer the current question directly. Use case numbers only when they appear above.
Use short headings and bullet points for complex answers. End personal medical
answers with a one-line reminder to confirm decisions with a qualified clinician."""
        resp = S.gemini_client.models.generate_content(
            model=S.gemini_model,
            contents=prompt,
            config=S.gemini_types.GenerateContentConfig(temperature=0.25),
        )
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            return {"answer": text, "source": "gemini", "model": S.gemini_model}

    if callable(S.tool4_answer):
        try:
            import inspect
            n = len([p for p in inspect.signature(S.tool4_answer).parameters.values()
                     if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
            out = S.tool4_answer(message, "Patient" if mode == "patient" else "Technical") if n >= 2 \
                else S.tool4_answer(message)
            if out:
                return {"answer": str(out), "source": "tool4", "model": "Tool 4 knowledge base"}
        except Exception:
            pass

    return {"answer": None, "source": "offline", "model": None}


# ----------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------

def _wrap_html(body):
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
            "</head><body>" + body + "</body></html>")


def create_app():
    from typing import List
    from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    from fastapi import Request
    app = FastAPI(title="NeuroNova", version=VERSION)

    def fail(status, msg):
        raise HTTPException(status_code=status, detail=msg)

    OPEN_PATHS = {"/api/health", "/api/auth/login", "/api/auth/logout", "/api/auth/me"}

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        user = None
        if S.store is not None:
            token = (request.cookies.get("nn_session") or request.headers.get("x-nn-token")
                     or request.query_params.get("nn_token"))
            if token:
                user = session_user(token)
        if not S.auth_enabled:
            user = user or _LOCAL_USER
        ctx = _CURRENT_USER.set(user)
        try:
            if S.auth_enabled and request.url.path.startswith("/api/") and request.url.path not in OPEN_PATHS and not user:
                return JSONResponse({"detail": "Please sign in.", "auth_required": True}, status_code=401)
            return await call_next(request)
        finally:
            _CURRENT_USER.reset(ctx)

    def need(*roles):
        u = current_user()
        if not u or (roles and u["role"] not in roles):
            fail(403, "Your role cannot do this. Sign in as " + (" or ".join(roles) if roles else "a user") + ".")
        return u

    def db():
        if S.store is None:
            fail(503, "The case database is not available in this session.")
        return S.store

    @app.post("/api/auth/login")
    def auth_login(payload: dict = Body(...)):
        st = db()
        name = str(payload.get("username") or "").strip().lower()
        res = login(name, str(payload.get("password") or ""))
        if not res:
            st.q("INSERT INTO audit (ts, username, role, action, target, detail) VALUES (?,?,?,?,?,?)",
                 (_now(), name[:40], "-", "auth.failed", name[:40], "wrong username or password"))
            fail(401, "Wrong username or password.")
        token, user = res
        _CURRENT_USER.set(user)
        audit("auth.login", user["username"])
        resp = JSONResponse({"user": user, "token": token})
        resp.set_cookie("nn_session", token, httponly=True, samesite="lax", max_age=SESSION_HOURS * 3600)
        return resp

    @app.post("/api/auth/logout")
    def auth_logout(request: Request):
        token = request.cookies.get("nn_session") or request.headers.get("x-nn-token")
        if token and S.store is not None:
            audit("auth.logout", (current_user() or {}).get("username", ""))
            S.store.q("DELETE FROM sessions WHERE token=?", (token,))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie("nn_session")
        return resp

    @app.get("/api/auth/me")
    def auth_me():
        return {"auth": S.auth_enabled, "user": current_user(), "database": S.store is not None}

    @app.get("/api/users")
    def users_list():
        need("admin")
        return {"users": db().q("SELECT username, name, role, active, created_at, last_login FROM users ORDER BY role, username")}

    @app.post("/api/users")
    def users_create(payload: dict = Body(...)):
        need("admin")
        name = re.sub(r"[^a-z0-9_.-]", "", str(payload.get("username") or "").strip().lower())[:32]
        role = payload.get("role")
        pw = str(payload.get("password") or "")
        if not name or role not in ROLES or len(pw) < 8:
            fail(400, "Give a username, a role (admin, doctor or researcher) and a password of at least 8 characters.")
        if db().q("SELECT 1 FROM users WHERE username=?", (name,), one=True):
            fail(409, "That username exists.")
        salt, h = _hash_pw(pw)
        db().q("INSERT INTO users (username, name, role, salt, pw_hash, active, created_at) VALUES (?,?,?,?,?,1,?)",
               (name, str(payload.get("name") or name)[:80], role, salt, h, _now()))
        audit("user.create", name, role)
        return {"ok": True}

    @app.post("/api/users/{username}")
    def users_update(username: str, payload: dict = Body(...)):
        me = need("admin")
        row = db().q("SELECT * FROM users WHERE username=?", (username,), one=True)
        if not row:
            fail(404, "No such user.")
        if "role" in payload:
            if payload["role"] not in ROLES:
                fail(400, "Unknown role.")
            if username == me["username"] and payload["role"] != "admin":
                fail(400, "You cannot remove your own admin role.")
            db().q("UPDATE users SET role=? WHERE username=?", (payload["role"], username))
        if "active" in payload:
            if username == me["username"] and not payload["active"]:
                fail(400, "You cannot deactivate yourself.")
            db().q("UPDATE users SET active=? WHERE username=?", (1 if payload["active"] else 0, username))
            if not payload["active"]:
                db().q("DELETE FROM sessions WHERE username=?", (username,))
        if payload.get("password"):
            if len(str(payload["password"])) < 8:
                fail(400, "Passwords need at least 8 characters.")
            salt, h = _hash_pw(str(payload["password"]))
            db().q("UPDATE users SET salt=?, pw_hash=? WHERE username=?", (salt, h, username))
        audit("user.update", username, json.dumps({k: (v if k != "password" else "***") for k, v in payload.items()}))
        return {"ok": True}

    @app.get("/api/patients")
    def patients_list(q: str = ""):
        need()
        rows = db().q("""SELECT p.*, (SELECT COUNT(*) FROM cases c WHERE c.patient_id = p.patient_id) AS cases
                         FROM patients p ORDER BY p.patient_id""")
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in r["patient_id"].lower() or ql in (r["name"] or "").lower()]
        return {"patients": [_pseudo(r) for r in rows]}

    @app.post("/api/patients")
    def patients_create(payload: dict = Body(...)):
        need("doctor", "admin")
        n = db().q("SELECT COUNT(*) AS n FROM patients", one=True)["n"]
        pid = f"PT-{n + 1:04d}"
        while db().q("SELECT 1 FROM patients WHERE patient_id=?", (pid,), one=True):
            n += 1
            pid = f"PT-{n + 1:04d}"
        try:
            age = int(payload.get("age")) if str(payload.get("age") or "").strip() else None
        except ValueError:
            fail(400, "Age must be a number.")
        db().q("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?)",
               (pid, str(payload.get("name") or "Unnamed")[:80], age, str(payload.get("sex") or "")[:12],
                str(payload.get("history") or "")[:2000], 1 if payload.get("synthetic", True) else 0, _now(),
                current_user()["username"]))
        audit("patient.create", pid)
        return {"patient_id": pid}

    @app.get("/api/patients/{pid}")
    def patient_get(pid: str):
        need()
        p = db().q("SELECT * FROM patients WHERE patient_id=?", (pid,), one=True)
        if not p:
            fail(404, "No such patient.")
        cases = db().q("SELECT * FROM cases WHERE patient_id=? ORDER BY created_at DESC", (pid,))
        return {"patient": _pseudo(p), "cases": cases}

    @app.get("/api/audit")
    def audit_list(limit: int = 200, action: str = ""):
        need("admin")
        if action:
            rows = db().q("SELECT * FROM audit WHERE action LIKE ? ORDER BY id DESC LIMIT ?", (action + "%", int(limit)))
        else:
            rows = db().q("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (int(limit),))
        return {"audit": rows}

    @app.post("/api/case/{case_id}/review")
    def case_review(case_id: str, payload: dict = Body(...)):
        need("doctor", "admin")
        try:
            row = review_case(case_id, payload.get("verdict"), payload.get("label"), str(payload.get("note") or ""))
        except KeyError as exc:
            fail(404, exc.args[0])
        except ValueError as exc:
            fail(400, str(exc))
        return {"record": row, "reviews": db().q("SELECT * FROM reviews WHERE case_id=? ORDER BY id DESC", (case_id,))}

    @app.get("/api/case/{case_id}/reviews")
    def case_reviews(case_id: str):
        need()
        return {"record": db().q("SELECT * FROM cases WHERE case_id=?", (case_id,), one=True),
                "reviews": db().q("SELECT * FROM reviews WHERE case_id=? ORDER BY id DESC", (case_id,))}

    @app.delete("/api/case/{case_id}")
    def case_delete(case_id: str):
        need("admin")
        db().q("DELETE FROM cases WHERE case_id=?", (case_id,))
        db().q("DELETE FROM reviews WHERE case_id=?", (case_id,))
        for ext in ("jpg", "json.gz"):
            try:
                os.remove(db().case_file(case_id, ext))
            except OSError:
                pass
        S.cases.pop(case_id, None)
        audit("case.delete", case_id)
        return {"ok": True}

    @app.get("/api/compare")
    def case_compare(a: str, b: str):
        need()
        try:
            return _finite(compare_cases(a, b))
        except KeyError as exc:
            fail(404, exc.args[0])

    @app.get("/api/case/{case_id}/pdf")
    def case_pdf_api(case_id: str):
        need()
        try:
            case = _case(case_id)
        except KeyError as exc:
            fail(404, exc.args[0])
        try:
            raw = case_pdf(case)
        except Exception as exc:
            fail(500, f"PDF failed: {type(exc).__name__}: {exc}")
        return Response(content=raw, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{case_id}_report.pdf"'})

    @app.get("/", response_class=HTMLResponse)
    def index():
        path = S.html_path
        if not path or not os.path.exists(path):
            return HTMLResponse("<h1>neuronova_index.html not found</h1><p>Run the cell that writes "
                                "/content/neuronova_index.html, then run the launch cell again.</p>", 500)
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(_wrap_html(f.read()))

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "version": VERSION,
            "classes": S.class_names,
            "models": {"efficientnet": S.effnet is not None, "inception": S.inception is not None,
                       "densenet": S.densenet is not None},
            "gemini": S.gemini_client is not None,
            "gemini_model": S.gemini_model if S.gemini_client is not None else None,
            "tool4": callable(S.tool4_answer),
            "samples": len(S.samples),
            "pipeline": PIPELINE,
            "index": S.index_status,
            "evaluation": S.eval_status,
            "testing_folder": S.test_dir is not None,
            "training_folder": S.train_dir is not None,
            "reliability": _finite(reliability_settings()),
            "auth": S.auth_enabled,
            "database": S.store is not None,
            "unet": ({"arch": (S.unet_meta or {}).get("arch", "U-Net"), "input": int(S.unet.input_shape[1]),
                      "test_dice": ((S.unet_meta or {}).get("test") or {}).get("dice")} if S.unet is not None else None),
        }

    @app.get("/api/samples")
    def samples():
        out = []
        for s in S.samples:
            try:
                im = Image.open(s["path"]).convert("RGB")
                im.thumbnail((160, 160))
                out.append({"id": s["id"], "label": s["label"], "thumb": _png_data_url(im, "JPEG")})
            except Exception:
                continue
        return {"samples": out}

    @app.post("/api/case")
    async def new_case(file: UploadFile = File(None), sample_id: str = Form(None), patient_id: str = Form(None)):
        ground_truth = None
        if sample_id:
            match = next((s for s in S.samples if s["id"] == sample_id), None)
            if match is None:
                fail(404, "That sample is not available. Pick one from the list.")
            raw = open(match["path"], "rb").read()
            filename = os.path.basename(match["path"])
            ground_truth = match["label"]
            source = "sample"
        elif file is not None:
            filename = os.path.basename(file.filename or "upload.png")
            if not filename.lower().endswith(ALLOWED_EXT + (".dcm",)):
                fail(415, "Use a JPG, PNG or DICOM (.dcm) image. For a 3D scan use the 3D MRI viewer.")
            raw = await file.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                fail(413, "The image is larger than 15 MB. Export a smaller JPG or PNG.")
            source = "upload"
        else:
            fail(400, "Attach an MRI image or choose a sample.")

        spacing, dicom_meta = None, None
        try:
            if filename.lower().endswith(".dcm"):
                img, spacing, dicom_meta = read_dicom_image(raw)
            else:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            fail(415, "The file could not be opened as an image.")

        case = new_case_from_image(img, filename, ground_truth, source)
        if spacing:
            case["pixel_spacing_mm"] = spacing
        if dicom_meta is not None:
            case["dicom"] = dicom_meta
        if patient_id and S.store is not None and S.store.q("SELECT 1 FROM patients WHERE patient_id=?",
                                                            (patient_id,), one=True):
            case["patient_id"] = patient_id
        try:
            with S.lock:
                case["results"]["classify"] = stage_classify(case)
        except Exception as exc:
            fail(500, f"Classification failed: {type(exc).__name__}: {exc}")
        _remember(case)
        try:
            persist_case(case, new=True)
        except Exception as exc:
            print("  could not save the case:", exc)
        return {
            "case_id": case["case_id"],
            "created_at": case["created_at"],
            "filename": filename,
            "source": source,
            "ground_truth": ground_truth,
            "image": case["image_payload"],
            "classify": case["results"]["classify"],
            "dicom": case.get("dicom"),
            "pixel_spacing_mm": case.get("pixel_spacing_mm"),
        }

    @app.post("/api/volume")
    async def new_volume(file: UploadFile = File(None), files: List[UploadFile] = File(None),
                         seg: UploadFile = File(None)):
        import tempfile
        ups = [f for f in ([file] if file is not None else []) + list(files or []) if f is not None]
        if not ups:
            fail(400, "Attach a .nii, .nii.gz, a .zip of a DICOM series, or several .dcm files.")
        tmp = tempfile.mkdtemp(prefix="neuronova_vol_")
        try:
            saved = []
            total = 0
            for u in ups:
                name = os.path.basename(u.filename or "volume.nii.gz")
                if not name.lower().endswith(VOLUME_EXT):
                    fail(415, f"{name}: use .nii, .nii.gz, .zip (DICOM series) or .dcm files.")
                raw = await u.read()
                total += len(raw)
                if total > 600 * 1024 * 1024:
                    fail(413, "The upload is larger than 600 MB.")
                path = os.path.join(tmp, name)
                with open(path, "wb") as fh:
                    fh.write(raw)
                saved.append(path)
            seg_path = None
            if seg is not None and seg.filename:
                seg_path = os.path.join(tmp, "gt_" + os.path.basename(seg.filename))
                with open(seg_path, "wb") as fh:
                    fh.write(await seg.read())
            dcm = [p for p in saved if p.lower().endswith(".dcm")]
            audit("volume.upload", os.path.basename(saved[0]), f"{len(saved)} file(s)")
            try:
                if len(dcm) > 1:
                    res = process_volume(paths=dcm, name=f"{len(dcm)} DICOM files", seg_path=seg_path)
                else:
                    res = process_volume(path=saved[0], name=os.path.basename(saved[0]), seg_path=seg_path)
            except (ValueError, KeyError) as exc:
                fail(422, str(exc))
            except Exception as exc:
                fail(500, f"Volume processing failed: {type(exc).__name__}: {exc}")
            return _finite(res)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.get("/api/volume/{vid}/slice")
    def get_volume_slice(vid: str, plane: str = "axial", i: int = None, overlay: int = 1):
        try:
            raw, n, idx = volume_slice(vid, plane, i, bool(overlay))
        except KeyError as exc:
            fail(404, exc.args[0])
        return Response(content=raw, media_type="image/jpeg", headers={"X-Count": str(n), "X-Index": str(idx),
                                                                        "Cache-Control": "max-age=600"})

    @app.post("/api/case/{case_id}/{stage}")
    def run_stage(case_id: str, stage: str):
        fn = STAGES.get(stage)
        if fn is None:
            fail(404, f"Unknown stage '{stage}'.")
        try:
            case = _case(case_id)
        except KeyError as exc:
            fail(404, exc.args[0] if exc.args else "Case not found.")
        try:
            if getattr(fn, "self_locking", False):
                result = fn(case)
            else:
                with S.lock:
                    result = fn(case)
        except HTTPException:
            raise
        except Exception as exc:
            fail(500, f"{stage} failed: {type(exc).__name__}: {exc}")
        result = _finite(result)
        case["results"][stage] = result
        try:
            persist_case(case)
        except Exception:
            pass
        return result

    @app.get("/api/cases")
    def list_cases(status: str = "", patient_id: str = "", q: str = "", limit: int = 200):
        if S.store is None:
            out = []
            for c in sorted(S.cases.values(), key=lambda c: -c["t"]):
                r = c["results"]
                top = r["hybrid"]["hybrid"] if "hybrid" in r else r.get("classify")
                th = c["image"].copy()
                th.thumbnail((72, 72))
                out.append({"case_id": c["case_id"], "created_at": c.get("created_at"), "filename": c["filename"],
                            "source": c.get("source"), "prediction": top["prediction"] if top else None,
                            "confidence": top["confidence"] if top else None,
                            "steps": [k for k in PIPELINE if k in r], "thumb": _png_data_url(th, "JPEG")})
            return {"cases": out}
        sql, args = "SELECT c.*, p.name AS patient_name FROM cases c LEFT JOIN patients p ON p.patient_id = c.patient_id WHERE 1=1", []
        if status == "review":
            sql += " AND c.status != 'reviewed' AND c.decision IS NOT NULL"
        elif status == "reviewed":
            sql += " AND c.status = 'reviewed'"
        elif status == "withheld":
            sql += " AND c.decision = 'abstain'"
        elif status == "corrected":
            sql += " AND c.review_verdict = 'correct'"
        if patient_id:
            sql += " AND c.patient_id = ?"
            args.append(patient_id)
        if q:
            sql += " AND (c.case_id LIKE ? OR c.filename LIKE ? OR c.prediction LIKE ?)"
            args += [f"%{q}%"] * 3
        sql += " ORDER BY c.created_at DESC LIMIT ?"
        args.append(int(limit))
        rows = [_pseudo(r) for r in S.store.q(sql, tuple(args))]
        for r in rows:
            r["steps"] = [k for k in str(r.get("steps") or "").split(",") if k]
        return {"cases": rows}

    @app.get("/api/case/{case_id}")
    def get_case(case_id: str):
        try:
            c = _case(case_id)
        except KeyError as exc:
            fail(404, exc.args[0] if exc.args else "Case not found.")
        rec = S.store.q("SELECT * FROM cases WHERE case_id=?", (case_id,), one=True) if S.store else None
        pat = S.store.q("SELECT * FROM patients WHERE patient_id=?", (c.get("patient_id"),), one=True) \
            if (S.store and c.get("patient_id")) else None
        return _finite({"case_id": c["case_id"], "created_at": c.get("created_at"), "filename": c["filename"],
                        "source": c.get("source"), "ground_truth": c.get("ground_truth"), "image": c["image_payload"],
                        "dicom": c.get("dicom"), "pixel_spacing_mm": c.get("pixel_spacing_mm"),
                        "patient": _pseudo(pat) if pat else None, "record": rec, "agents": c.get("agents"),
                        **c["results"]})

    @app.get("/api/evaluation")
    def evaluation_status():
        return _finite({"status": S.eval_status, "result": S.eval_result if S.eval_status.get("state") == "done" else None,
                        "testing_folder": S.test_dir is not None})

    @app.post("/api/evaluation")
    def evaluation_start(payload: dict = Body(default={})):
        if S.test_dir is None:
            fail(409, "The Testing folder is not unzipped. Run the dataset and unzip cells in Colab first.")
        _start_eval_thread(force=bool((payload or {}).get("force")))
        audit("evaluation.run", "Testing folder")
        return {"status": S.eval_status}

    @app.get("/api/segmentation")
    def segmentation_info():
        meta = dict(S.unet_meta or {})
        meta.pop("history_full", None)
        return _finite({"available": S.unet is not None, "meta": meta or None, "xai_vs_masks": S.xai_mask_eval})

    @app.get("/api/models")
    def models_card():
        return _finite(model_info())

    @app.post("/api/batch")
    async def batch(files: List[UploadFile] = File(...)):
        if len(files) > 40:
            fail(413, "Send at most 40 images at a time.")
        pils, names, rows_err = [], [], []
        for f in files:
            name = os.path.basename(f.filename or "image")
            raw = await f.read()
            if not name.lower().endswith(ALLOWED_EXT) or len(raw) > MAX_UPLOAD_BYTES:
                rows_err.append({"file": name, "error": "Use a JPG or PNG under 15 MB."})
                continue
            try:
                pils.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                names.append(name)
            except Exception:
                rows_err.append({"file": name, "error": "Could not open this file as an image."})
        try:
            rows = batch_predict(pils, names)
        except Exception as exc:
            fail(500, f"Batch analysis failed: {type(exc).__name__}: {exc}")
        audit("batch.run", f"{len(names)} images")
        return _finite({"rows": rows + rows_err})

    @app.get("/api/rag")
    def rag_info():
        R = S.rag or {}
        return {"status": S.rag_status, "docs": R.get("docs", []), "method": R.get("method"),
                "folder": R.get("folder"), "gemini": S.gemini_client is not None}

    @app.post("/api/rag/reindex")
    def rag_reindex():
        need("doctor", "admin")
        if S.rag_status.get("state") == "building":
            return {"status": S.rag_status}
        _start_rag_thread()
        return {"status": S.rag_status}

    @app.post("/api/rag/upload")
    async def rag_upload(file: UploadFile = File(...)):
        need("doctor", "admin")
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", os.path.basename(file.filename or "document.pdf"))
        if not name.lower().endswith(RAG_EXT):
            fail(415, "Add PDF, TXT or Markdown documents.")
        raw = await file.read()
        if len(raw) > 40 * 1024 * 1024:
            fail(413, "Documents must be under 40 MB.")
        with open(os.path.join(_rag_dir(), name), "wb") as fh:
            fh.write(raw)
        audit("document.upload", name, f"{len(raw) // 1024} KB")
        _start_rag_thread()
        return {"saved": name, "status": S.rag_status}

    @app.post("/api/rag/search")
    def rag_search_api(payload: dict = Body(...)):
        q = str(payload.get("query") or "").strip()
        if not q:
            fail(400, "Type a question first.")
        if S.rag is None:
            fail(409, "The knowledge base is still being built. Try again in a moment.")
        return _finite({"hits": rag_search(q, int(payload.get("k") or 5))})

    @app.post("/api/rag/ask")
    def rag_ask(payload: dict = Body(...)):
        q = str(payload.get("question") or "").strip()
        if not q:
            fail(400, "Type a question first.")
        if S.rag is None:
            fail(409, "The knowledge base is still being built. Try again in a moment.")
        case = S.cases.get(payload.get("case_id") or "")
        try:
            return _finite(rag_answer(q, case))
        except Exception as exc:
            fail(500, f"Answer failed: {type(exc).__name__}: {exc}")

    @app.post("/api/agents/run")
    async def agents_run(case_id: str = Form(...), report_text: str = Form(""), file: UploadFile = File(None)):
        try:
            case = _case(case_id)
        except KeyError as exc:
            fail(404, exc.args[0] if exc.args else "Case not found.")
        text = report_text or ""
        if file is not None and file.filename:
            raw = await file.read()
            if len(raw) > 20 * 1024 * 1024:
                fail(413, "The report file must be under 20 MB.")
            if file.filename.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    text += "\n" + "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(raw)).pages)
                except Exception as exc:
                    fail(415, f"Could not read the PDF: {type(exc).__name__}")
            else:
                text += "\n" + raw.decode("utf-8", errors="ignore")
        try:
            res = run_agents(case, text[:20000])
            audit("agents.run", case_id, f"{len(res['findings'])} findings")
            persist_case(case)
        except ValueError as exc:
            fail(409, str(exc))
        except Exception as exc:
            fail(500, f"Agent workflow failed: {type(exc).__name__}: {exc}")
        return _finite(res)

    @app.get("/api/research")
    def research_get():
        return _finite({"results": S.research, "status": S.lab_status,
                        "folders": {"training": S.train_dir is not None, "testing": S.test_dir is not None}})

    @app.post("/api/research/{kind}")
    def research_run(kind: str, payload: dict = Body(default={})):
        fns = {"federated": lambda: federated_experiment(verbose=False),
               "domain_shift": lambda: domain_shift_test(verbose=False),
               "faithfulness": lambda: faithfulness_benchmark(verbose=False)}
        if kind not in fns:
            fail(404, "Unknown experiment.")
        if (S.lab_status.get(kind) or {}).get("state") == "running":
            return {"status": S.lab_status[kind]}
        user = current_user()

        def work():
            tok = _CURRENT_USER.set(user)
            try:
                fns[kind]()
            except Exception as exc:
                _set_lab(kind, state="error", phase="", message=f"{type(exc).__name__}: {exc}")
            finally:
                _CURRENT_USER.reset(tok)
        _set_lab(kind, state="running", phase="starting", message=None)
        threading.Thread(target=work, daemon=True).start()
        audit("research.run", kind)
        return {"status": S.lab_status[kind]}

    @app.get("/api/monitoring")
    def monitoring_get(window: int = 50):
        return _finite(monitoring_report(window))

    @app.post("/api/chat")
    def chat(payload: dict = Body(...)):
        message = str(payload.get("message") or "").strip()
        if not message:
            fail(400, "Type a question first.")
        case = S.cases.get(payload.get("case_id") or "")
        mode = "patient" if payload.get("mode") == "patient" else "research"
        try:
            return chat_answer(message[:4000], payload.get("history") or [], case, mode,
                               payload.get("patient") or {})
        except Exception as exc:
            return JSONResponse({"answer": None, "source": "offline", "model": None,
                                 "error": f"{type(exc).__name__}: {exc}"})

    return app


# ----------------------------------------------------------------
# Launch helpers (Colab proxy + optional public link)
# ----------------------------------------------------------------

def _start_server(app, port):
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.1)
    return server, th


def _public_tunnel(port, timeout=40):
    """Cloudflare quick tunnel: free https link, no account. Anyone with the link can use the app."""
    exe = shutil.which("cloudflared") or "/content/cloudflared"
    if not os.path.exists(exe):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        subprocess.run(["wget", "-q", "-O", exe, url], check=True)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_end = time.time() + timeout
    while time.time() < t_end:
        line = proc.stdout.readline()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
        if m:
            return m.group(0), proc
    return None, proc


def _start_rag_thread():
    def work():
        try:
            build_rag_index(verbose=False)
        except Exception as exc:
            S.rag_status = {"state": "error", "message": f"{type(exc).__name__}: {exc}"}
    S.rag_status = {"state": "building"}
    threading.Thread(target=work, daemon=True).start()


def _setup_train_dir():
    S.train_dir = next((d for d in TRAIN_DIR_CANDIDATES if os.path.isdir(d)), None)
    if S.train_dir:
        print(f"  Training scans    : {S.train_dir} (similar-case index builds in the background)")
    else:
        print("  Training scans    : not unzipped (Tool 8A feature check and 8C similar cases need them)")


def setup(ns=None, html_path=None, auth=None):
    """Load models and data without starting the web server (used by launch() and the Tool 8 cells)."""
    ns = ns if ns is not None else {}
    S.html_path = html_path or next(
        (p for p in ["/content/neuronova_index.html", os.path.join(os.getcwd(), "neuronova_index.html"),
                     os.path.join(_HERE, "neuronova_index.html")] if os.path.exists(p)), None)
    _setup_models(ns)
    _setup_unet()
    _setup_gemini(ns)
    try:
        _setup_store(S.auth_enabled if auth is None else auth)
    except Exception as exc:
        print("  Case records      : database unavailable:", exc)
        S.store = None
    S.tool4_answer = ns.get("answer_question") if callable(ns.get("answer_question")) else None
    _setup_samples()
    _setup_train_dir()
    S.index, S.eval_result = None, None
    S.index_status, S.eval_status = {"state": "idle"}, {"state": "idle"}
    _load_cached_evaluation()
    if S.eval_result:
        print("  Evaluation        : cached Tool 8D results found")
    _start_index_thread()
    _start_rag_thread()
    S.lab_status = {}
    _load_research()
    S.ready = True
    return S


def launch(ns=None, port=8765, html_path=None, public=False, inline=False, height=900, auth=True):
    """Start NeuroNova. Pass globals() from the notebook so it reuses your loaded models."""
    ns = ns if ns is not None else {}
    print("=" * 78)
    print(f"  NEURONOVA {VERSION} — starting web interface")
    print("=" * 78)
    setup(ns, html_path, auth=auth)
    if not S.html_path:
        print("  WARNING: neuronova_index.html not found. Copy it to /content first.")

    old = ns.get("_neuronova_server")
    if old is not None:
        try:
            old.should_exit = True
            time.sleep(1.2)
        except Exception:
            pass

    server, _ = _start_server(create_app(), port)
    ns["_neuronova_server"] = server
    if not getattr(server, "started", False):
        print(f"\n  Server did not start on port {port}. Is another program using it? Try launch(globals(), port={port + 1}).")
        return None

    print(f"\n  Server running on port {port}")
    in_colab = "google.colab" in sys.modules
    if in_colab:
        from google.colab import output
        try:
            from google.colab.output import eval_js
            url = eval_js(f"google.colab.kernel.proxyPort({port})")
            print("  Open in a new tab :", url)
        except Exception:
            pass
        if inline:
            output.serve_kernel_port_as_iframe(port, height=height)
        else:
            output.serve_kernel_port_as_window(port, anchor_text="Open NeuroNova")
    else:
        print(f"  Open http://localhost:{port}")

    if public:
        try:
            url, proc = _public_tunnel(port)
            ns["_neuronova_tunnel"] = proc
            print("  Public link       :", url or "could not start (see cloudflared output)")
            print("  Anyone with this link can upload images to your Colab session.")
        except Exception as exc:
            print("  Public link failed:", exc)

    print("\n  Research and education use only. Not a medical device.")
    print("=" * 78)
    return None     # the server is kept in ns["_neuronova_server"]; returning nothing keeps the cell output clean


if __name__ == "__main__":
    # Standalone: python neuronova_server.py  (expects the .keras files next to it or in /content)
    srv = launch({}, port=int(os.environ.get("PORT", "8765")))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.should_exit = True
