# 🧠 Visual Question Answering (VQA) System

> Upload an image, ask a natural language question, and compare real-time inference across multiple CNN-LSTM architectures.

🌐 **Live Demo:** [https://eba38cprwk.us-east-1.awsapprunner.com/](https://eba38cprwk.us-east-1.awsapprunner.com/)

---

## 📌 Overview

This project implements a **Visual Question Answering** system that integrates visual perception (CNN-based image encoding) with natural language understanding (LSTM-based question encoding) to generate free-form text answers. 

Four model variants are trained and compared systematically, exploring the effects of **pretrained vision backbones** and **spatial attention mechanisms**. The entire system is deployed as a web application on **AWS App Runner** using Docker.

---

## 🌐 Web Application

| Section | Description |
|---|---|
| **Image Upload** | Drag & drop or click to upload JPG / PNG / WEBP |
| **Question Input** | Free-form natural language question |
| **Run Analysis** | Runs inference across all 4 model variants simultaneously |
| **Model Comparisons** | Side-by-side predictions from each architecture |

---

## 🏗️ Model Architectures

The project trains and compares **4 model variants** (M1–M4):

| ID | Name | CNN Backbone | Attention | Description |
|---|---|---|---|---|
| **M1** | `M1_Scratch_NoAttn` | Trained from scratch | ❌ | Baseline CNN-LSTM |
| **M2** | `M2_Scratch_Attn` | Trained from scratch | ✅ Spatial | Attention CNN-LSTM |
| **M3** | `M3_Pretrained_NoAttn` | ResNet-18 (ImageNet) | ❌ | Pre-trained Vision |
| **M4** | `M4_Pretrained_Attn` | ResNet-18 (ImageNet) | ✅ Spatial | **State of the Art** |

### Architecture Components

```
Image ──► CNN Encoder (ResNet-18 / Scratch)
                    │
                    ▼
Question ──► Question LSTM (2-layer, hidden=256)
                    │
                    ▼
           [Spatial Attention] (optional)
                    │
                    ▼
           Answer LSTM Decoder
                    │
                    ▼
           Beam Search Generation
```

- **Embedding:** GloVe 300d vectors
- **LSTM:** 2-layer, hidden size 256, dropout 0.3
- **Decoding:** Beam search (width=10) with length penalty & repetition penalty
- **Training:** Teacher forcing with cosine annealing LR schedule

---

## 📂 Project Structure

```
VQA/
├── VQA.ipynb                      # Main training & analysis notebook
├── Dockerfile                     # Container configuration
├── requirements.txt               # Python dependencies
├── vqa_deploy_all_models.pth     # Deployment artifact (all 4 models)
│
├── configs/
│   └── default.yaml              # Training hyperparameters & model config
│
├── src/                          # Core source code
│   ├── config.py                  # Config loader
│   ├── data/
│   │   ├── dataset.py             # A-OKVQA dataset class
│   │   ├── preprocessing.py       # Text tokenization & vocab building
│   │   └── glove.py               # GloVe embedding loader
│   ├── models/
│   │   ├── vqa_model.py           # Main VQAModel (forward + beam search)
│   │   ├── encoder.py             # CNN + Question LSTM encoders
│   │   ├── decoder.py             # Answer LSTM decoder
│   │   └── attention.py           # Spatial attention module
│   ├── engine/                    # Training & evaluation loops
│   └── utils/                     # Helpers & visualization tools
│
├── web/                           # FastAPI web backend
│   ├── main.py                    # API server (FastAPI)
│   └── static/                   # Frontend (HTML/CSS/JS)
│
├── checkpoints/                   # Saved model checkpoints per variant
├── logs/                          # Training logs
└── figures/                       # Generated plots & visualizations
```

---

## 📊 Dataset

**A-OKVQA** (A Benchmark for Visual Question Answering using World Knowledge)

| Property | Value |
|---|---|
| Source | `HuggingFaceM4/A-OKVQA` (auto-downloaded) |
| Split | 85% train / 15% validation |
| Rationale expansion | 3× (all rationale annotations used) |
| Image size | 224 × 224 (ResNet-compatible) |
| Min word frequency | 3 (vocabulary filtering) |

---

## ⚙️ Configuration

Key hyperparameters defined in [`configs/default.yaml`](configs/default.yaml):

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Optimizer | Adam + Cosine Annealing |
| Label smoothing | 0.1 |
| Gradient clipping | 5.0 |
| Early stopping patience | 7 epochs |
| Warmup epochs | 3 |
| Teacher forcing | 1.0 → 0.0 (annealed) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional but recommended)

### 1. Clone & Install

```bash
git clone <repo-url>
cd VQA
pip install -r requirements.txt
```

### 2. Train Models (Jupyter Notebook)

Open and run `VQA.ipynb` end-to-end. The notebook will:
1. Download the A-OKVQA dataset automatically via HuggingFace
2. Build question & answer vocabularies
3. Train all 4 model variants
4. Save checkpoints to `checkpoints/`
5. Export the deployment artifact `vqa_deploy_all_models.pth`

### 3. Run the Web Server Locally

```bash
uvicorn web.main:app --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000`

### 4. Run with Docker

```bash
docker build -t vqa-system .
docker run -p 8000:8000 vqa-system
```

---

## 🔌 API Reference

### `POST /v1/predict`

Run VQA inference for an uploaded image and question.

**Form Data:**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File | ✅ | JPEG / PNG / WEBP image |
| `question` | string | ✅ | Natural language question |
| `model_name` | string | ❌ | One of `M1_Scratch_NoAttn`, `M2_Scratch_Attn`, `M3_Pretrained_NoAttn`, `M4_Pretrained_Attn`. Leave empty to run all. |

**Response:**

```json
{
  "question": "What is the person doing?",
  "answers": {
    "M1_Scratch_NoAttn": "riding a bicycle",
    "M2_Scratch_Attn": "riding a bicycle in the park",
    "M3_Pretrained_NoAttn": "cycling down the road",
    "M4_Pretrained_Attn": "riding a bike along the path"
  },
  "models": ["M1_Scratch_NoAttn", "M2_Scratch_Attn", "M3_Pretrained_NoAttn", "M4_Pretrained_Attn"]
}
```

### `GET /health`

```json
{ "status": "ok", "num_models": "4" }
```

---

## 📈 Visualizations

The notebook generates the following analysis figures (saved in `figures/`):

| Figure | Description |
|---|---|
| `fig_training_1_loss_lr.png` | Training loss & learning rate curves |
| `fig5_bar.png` | F1 score bar chart per model |
| `fig4_radar.png` | Radar comparison of all 4 models |
| `fig6_attention_heatmap.png` | Spatial attention heatmaps |
| `fig7_attention_overlay.png` | Attention overlaid on original images |
| `fig10_confusion_matrix.png` | Prediction confusion matrix |
| `fig9_question_type.png` | Performance breakdown by question type |

---

## 🐳 Deployment

The application is containerized with **Docker** and deployed on **AWS App Runner**:

- **Base image:** `python:3.9-slim`
- **Server:** [Uvicorn](https://www.uvicorn.org/) + [FastAPI](https://fastapi.tiangolo.com/)
- **Port:** 8000
- **Model loading:** All 4 models are loaded at startup from `vqa_deploy_all_models.pth` (~344 MB)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch >= 2.0.0` | Deep learning framework |
| `torchvision >= 0.15.0` | Image transforms & ResNet backbone |
| `fastapi` | REST API server |
| `datasets >= 2.14.0` | HuggingFace dataset loading (A-OKVQA) |
| `numpy`, `Pillow` | Data processing |
| `nltk >= 3.8.0` | NLP metrics (BLEU, etc.) |
| `matplotlib >= 3.7.0` | Visualization |
| `pyyaml >= 6.0` | Configuration management |
| `scipy >= 1.10.0` | Statistical significance testing |

---

## 👥 Authors

| Name | Student ID | Contact |
|---|---|---|
| Member 1 | 523H0178 | 523H0178@gmail.com |
| Member 2 | 523H0173 | 523H0173@gmail.com |

---

## 📄 License

This project is developed for academic purposes as part of a Deep Learning course project.
