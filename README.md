# NeuroNova: brain tumour MRI analysis and explainability

A Google Colab project that classifies brain MRI slices into glioma, meningioma, pituitary tumour or no tumour, explains each prediction, and checks how far the answer can be trusted. It ends in **NeuroNova**, a web app that runs inside Colab.

> Research and education use only. Not a medical device. Use synthetic or properly anonymised data only.

## Files

| File | What it is |
|---|---|
| `project_tools_hybrid_16.ipynb` | The notebook: data, training, Tools 1–14, Tools 4–6 and the Tool 7 web app |
| `neuronova_server.py` | The backend: models, reliability, segmentation, 3D, RAG, agents, case records, research lab |
| `neuronova_index.html` | The NeuroNova web page |

## What the notebook does

| Tool | What it adds |
|---|---|
| **1–3D** | EfficientNetV2-B0 and InceptionV3 classification, Grad-CAM, the hybrid vote, a stability test (test-time augmentation), occlusion sensitivity and the XAI bridge |
| **8** Reliability | Monte Carlo dropout uncertainty, input validation (image quality and "is this a brain MRI?"), temperature scaling, a reject option, full evaluation (confusion matrix, ROC and precision-recall, calibration, speed) and batch analysis |
| **9** Third model | DenseNet121 trained with a proper validation split (optional, needs a GPU) |
| **10** Segmentation | U-Net / Attention U-Net tumour masks with Dice and IoU, and a test of the heatmaps against expert masks |
| **11** 3D MRI | NIfTI or DICOM volumes: slice-by-slice segmentation, tumour volume in cm³ and a rotatable 3D view |
| **12** Research lab | Federated learning (FedAvg), domain-shift robustness, explanation faithfulness and drift monitoring |
| **13** Medical RAG | Answers from trusted documents with page citations and answer checking |
| **14** Multi-agent workflow | Image, XAI, report-text, evidence, validation and writer agents with a final safety check |
| **4, 5, 6** | Explainable medical agent, case reports and the Gemini agentic assistant |
| **7** NeuroNova | The web app with sign-in and roles, saved case records, doctor review, PDF reports and an audit log |

## Run it in Google Colab

1. Put `neuronova_server.py` and `neuronova_index.html` in Google Drive, in `MyDrive/brain_tumor_models`.
2. Open `project_tools_hybrid_16.ipynb` in Colab and run the dataset cell.
3. Run the cells in order. The Tools 8–14 setup cell also scores the Testing folder once, so later cells use calibrated confidence and the reject threshold. Tools 9, 10 and 11 need extra data or a GPU and can be skipped.
4. Run Tool 7 last. It prints the demo sign-in accounts (admin, doctor, researcher) and saves them in `brain_tumor_models/neuronova_data/demo_logins.txt`.

### Optional data (in `MyDrive/brain_tumor_models`)

- `segmentation_dataset.zip` for Tool 10: a dataset **with tumour masks**, for example the Figshare brain tumour dataset (3,064 slices with masks), a copy of it with `images/` and `masks/` folders, or the LGG MRI segmentation set (`*_mask.tif`).
- `sample_volume.nii.gz` for Tool 11: one 3D scan, such as a BraTS case, or a `.zip` of a DICOM series. Its `*_seg.nii.gz` expert mask is optional.
- `medical_docs/` for Tool 13: trusted PDFs, `.txt` or `.md` files. A built-in NeuroNova handbook is always included.

### Gemini API key

Add `GEMINI_API_KEY` in Colab Secrets (the key icon) and enable notebook access. The key is read with `userdata.get` and is never stored in the notebook or sent to the web page. Without a key, the assistant, RAG and agents use extractive answers from the documents.

### Timings on a CPU runtime (first run; later runs load from Drive)

- Setup: about 5–8 min (search index and evaluation)
- Tool 9 DenseNet121: needs a GPU (15–25 min on a T4)
- Tool 10 U-Net: 30–40 min
- Tool 12 lab: a few minutes per experiment
