# 🌸 Flower Disease Detection — FedProx Federated Learning

EfficientNet-B1 trained with the **FedProx** federated learning algorithm on
a 6-class flower disease dataset. Targets **≥ 99% test accuracy**.

> **Device**: Runs on CPU (AMD Ryzen 5 5600H). No GPU/CUDA required.

---

## Classes

| Label | Disease |
|---|---|
| `Chrysanthemum_Bacterial_Leaf_Spot` | Bacterial leaf spot |
| `Chrysanthemum_Healthy` | Healthy |
| `Chrysanthemum_Septoria_Leaf_Spot` | Septoria leaf spot |
| `Jasmine_Healthy` | Healthy |
| `Jasmine_Multiple` | Multiple diseases |
| `Jasmine_Rust` | Rust disease |

---

## Project Structure

```
Plant_disease/
├── fedprox.py                     ← Main FedProx FL training script
├── flower_disease_efficientnet_gpu (1).py  ← Original centralised trainer
├── app.py                         ← Flask inference app
├── requirements.txt               ← Python dependencies
├── .env.example                   → Copy to .env and configure
├── .gitignore
├── dataset_flat_structure/        ← Dataset (not tracked by git)
│   ├── train/
│   ├── validation/
│   └── test/
├── checkpoints/                   ← Saved model weights
└── results/                       ← Plots, reports, history
```

---

## Quick Start

### 1. Clone / navigate to the project
```powershell
cd "d:\Plant Disease\Plant_disease"
```

### 2. Create & activate virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch (CPU)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies
```powershell
pip install -r requirements.txt
```

### 5. Set up environment config
```powershell
copy .env.example .env
# Edit .env if needed (paths, hyperparams, FL settings)
```

### 6. Run FedProx training
```powershell
python fedprox.py
```

---

## FedProx Algorithm

```
Global Model (Server)
      │
 ┌────┼────┐
 ▼    ▼    ▼
C1   C2   C3   ← 3 clients, 5 local epochs each
 │    │    │
 └────┴────┘
  FedAvg (weighted)
      │
Global Model (t+1)
```

**Local client loss:**
```
L = L_CrossEntropy + (μ/2) ‖w_local − w_global‖²
```

The proximal term (μ = 0.01) prevents client models from drifting too far
from the global model, enabling stable convergence across heterogeneous data.

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Model | EfficientNet-B1 (240×240) |
| Clients | 3 (IID partition) |
| Global rounds | 25 |
| Local epochs / round | 5 |
| μ (proximal) | 0.01 |
| Batch size | 16 |
| Learning rate | 3e-4 (OneCycleLR) |
| Augmentation | CLAHE · MixUp · CutMix · AutoAugment · TTA×5 |
| Backbone unfreeze | After round 8 |
| Early stopping | Patience = 10 |

All parameters are configurable via `.env`.

---

## Output Files

| File | Description |
|---|---|
| `checkpoints/best_fedprox_b1.pth` | Best global model weights |
| `results/training_curves_fedprox.png` | Val loss & accuracy per round |
| `results/confusion_matrix_test.png` | Test confusion matrix |
| `results/classification_report_test.txt` | Per-class precision/recall/F1 |
| `results/history_fedprox.json` | Raw training history |

---

## Inference (single image)

```python
import torch
from fedprox import predict_single, FL_CONFIG

ckpt = torch.load("checkpoints/best_fedprox_b1.pth", map_location="cpu")
# load model, then:
predict_single(model, "path/to/image.jpg",
               ckpt["class_names"], FL_CONFIG["model_version"],
               torch.device("cpu"))
```

---

## Estimated Training Time

| Hardware | Time/round | Total (25 rounds) |
|---|---|---|
| AMD Ryzen 5 5600H (CPU) | ~2–3 min | ~60–75 min |
| NVIDIA GPU (CUDA) | ~15–30 sec | ~7–12 min |

---

## Requirements

- Python 3.9+
- PyTorch 2.1+ (CPU)
- See `requirements.txt` for full list
