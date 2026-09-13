"""
Flower Disease Classification — FedProx Federated Learning
════════════════════════════════════════════════════════════════
Dataset : dataset_flat_structure/  (train / validation / test)
Classes : Chrysanthemum_Bacterial_Leaf_Spot, Chrysanthemum_Healthy,
          Chrysanthemum_Septoria_Leaf_Spot,
          Jasmine_Healthy, Jasmine_Multiple, Jasmine_Rust

FedProx Federated Learning
──────────────────────────
 • FedProx algorithm: local CE loss + (μ/2)‖w - w_global‖² proximal term
 • 3 clients, 25 global rounds, 5 local epochs each
 • Progressive backbone unfreezing after round 8
 • Weighted FedAvg aggregation
 • Full augmentation: CLAHE, SharpenFilter, MixUp, CutMix, AutoAugment
 • TTA × 5 at evaluation time

Device  : AMD Ryzen 5 5600H CPU (no CUDA — AMD GPU not supported by
          PyTorch on Windows).  All tensors live on CPU.
"""

import os
import gc
import time
import copy
import json
import warnings
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
#  DEVICE SETUP  (CPU-first; CUDA if available)
# ══════════════════════════════════════════════════════
def setup_device():
    """
    Detects available compute device.
    Prioritises CUDA, falls back to CPU with a clear banner.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.enabled     = True
        torch.backends.cudnn.benchmark   = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024 ** 3
        print("\n" + "═" * 60)
        print("  CUDA GPU DETECTED")
        print(f"  GPU       : {props.name}")
        print(f"  VRAM      : {vram:.1f} GB")
        print(f"  CUDA ver  : {torch.version.cuda}")
        print("═" * 60 + "\n")
    else:
        device = torch.device("cpu")
        print("\n" + "═" * 60)
        print("  RUNNING ON CPU  (no CUDA GPU detected)")
        print(f"  PyTorch : {torch.__version__}")
        print(f"  Threads : {torch.get_num_threads()}")
        print("  Tip: AMD Radeon GPUs require ROCm (Linux only).")
        print("  Training will use all CPU cores automatically.")
        print("═" * 60 + "\n")
        # Maximise CPU parallelism
        torch.set_num_threads(os.cpu_count() or 4)

    return device


# ══════════════════════════════════════════════════════
#  HELPER — parse env values
# ══════════════════════════════════════════════════════
def _env(key, default):
    val = os.getenv(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.strip().lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


# ══════════════════════════════════════════════════════
#  FEDERATED LEARNING CONFIG
# ══════════════════════════════════════════════════════
FL_CONFIG = {
    # ── Paths ──────────────────────────────────────
    "data_dir":    _env("DATA_DIR",    "./dataset_flat_structure"),
    "save_dir":    _env("SAVE_DIR",    "./checkpoints"),
    "results_dir": _env("RESULTS_DIR", "./results"),

    # ── Model ──────────────────────────────────────
    # B1 chosen for CPU: 240×240, ~60% fewer FLOPs than B3
    "model_version": _env("MODEL_VERSION", "b1"),

    # ── Standard training hyperparams ──────────────
    "batch_size":      _env("BATCH_SIZE",      16),
    "learning_rate":   _env("LEARNING_RATE",   3e-4),
    "weight_decay":    _env("WEIGHT_DECAY",    1e-4),
    "label_smoothing": _env("LABEL_SMOOTHING", 0.1),
    "dropout_rate":    _env("DROPOUT_RATE",    0.35),

    # ── Augmentation ───────────────────────────────
    "mixup_alpha": _env("MIXUP_ALPHA", 0.3),
    "cutmix_prob": _env("CUTMIX_PROB", 0.4),

    # ── Transfer-learning unfreezing ───────────────
    "unfreeze_last_n_blocks": _env("UNFREEZE_LAST_N_BLOCKS", 5),

    # ── Early stopping (based on val_acc) ──────────
    "patience": _env("PATIENCE", 10),

    # ── DataLoader (CPU-safe flags) ─────────────────
    "num_workers":        _env("NUM_WORKERS",        4),
    "pin_memory":         False,   # CPU training — pin_memory off
    "prefetch_factor":    _env("PREFETCH_FACTOR", 2),
    "persistent_workers": _env("PERSISTENT_WORKERS", True),

    # ── AMP — disabled on CPU ───────────────────────
    "use_amp": False,

    # ── Multi-GPU — not applicable ──────────────────
    "use_multi_gpu": False,

    # ── torch.compile — skip on CPU ─────────────────
    "compile_model": False,

    # ── TTA ────────────────────────────────────────
    "tta_n": _env("TTA_N", 5),

    # ── Reproducibility ────────────────────────────
    "seed": _env("SEED", 42),

    # ══ FedProx-specific ═══════════════════════════
    # Number of simulated clients
    "num_clients":   _env("NUM_CLIENTS",   3),
    # Global communication rounds
    "num_rounds":    _env("NUM_ROUNDS",    25),
    # Local epochs per client per round
    "local_epochs":  _env("LOCAL_EPOCHS",  5),
    # Proximal regularisation coefficient μ
    "mu":            _env("MU",            0.01),
    # Fraction of clients selected per round (1.0 = all)
    "fraction_fit":  _env("FRACTION_FIT",  1.0),
    # Unfreeze backbone after this round number
    "unfreeze_round": _env("UNFREEZE_ROUND", 8),
    # IID partition (True) or Non-IID Dirichlet (False)
    "iid_partition": _env("IID_PARTITION", True),
    # Dirichlet α for non-IID (lower = more heterogeneous)
    "dirichlet_alpha": _env("DIRICHLET_ALPHA", 0.5),
}

EFFICIENTNET_INPUT_SIZES = {
    "b0": 224, "b1": 240, "b2": 260, "b3": 300, "b4": 380,
}

EFFICIENTNET_REGISTRY = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    "b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
}


# ══════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ══════════════════════════════════════════════════════
#  CUSTOM PREPROCESSING TRANSFORMS
# ══════════════════════════════════════════════════════
class CLAHE:
    """Contrast Limited Adaptive Histogram Equalisation in LAB space."""
    def __init__(self, clip_limit=2.5, tile=(8, 8), p=0.5):
        self.clip_limit = clip_limit
        self.tile = tile
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        try:
            import cv2
            arr = np.array(img)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                     tileGridSize=self.tile)
            lab[..., 0] = clahe.apply(lab[..., 0])
            return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        except ImportError:
            return img


class SharpenFilter:
    """UnsharpMask — makes disease spots more prominent."""
    def __init__(self, radius=1.5, percent=130, threshold=2, p=0.4):
        self.radius = radius; self.percent = percent
        self.threshold = threshold; self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        return img.filter(ImageFilter.UnsharpMask(
            self.radius, self.percent, self.threshold))


class RandomBrightnessContrast:
    """Independent brightness + contrast jitter."""
    def __init__(self, br=(0.70, 1.30), cr=(0.70, 1.30), p=0.5):
        self.br = br; self.cr = cr; self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        img = ImageEnhance.Brightness(img).enhance(np.random.uniform(*self.br))
        img = ImageEnhance.Contrast(img).enhance(np.random.uniform(*self.cr))
        return img


class RandomGridShuffle:
    """Shuffles image grid tiles — forces local texture learning."""
    def __init__(self, grid=3, p=0.15):
        self.grid = grid; self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        w, h = img.size
        sw, sh = w // self.grid, h // self.grid
        tiles = [img.crop((j*sw, i*sh, (j+1)*sw, (i+1)*sh))
                 for i in range(self.grid) for j in range(self.grid)]
        np.random.shuffle(tiles)
        out = Image.new(img.mode, img.size)
        for idx, tile in enumerate(tiles):
            i, j = divmod(idx, self.grid)
            out.paste(tile, (j*sw, i*sh))
        return out


# ══════════════════════════════════════════════════════
#  DATA TRANSFORMS & LOADERS
# ══════════════════════════════════════════════════════
def build_transforms(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0),
                                     ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=35),
        CLAHE(clip_limit=2.5, tile=(8, 8), p=0.5),
        SharpenFilter(radius=1.5, percent=130, threshold=2, p=0.4),
        RandomBrightnessContrast(br=(0.70, 1.30), cr=(0.70, 1.30), p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.12),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.35, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.85, 1.15), shear=10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        RandomGridShuffle(grid=3, p=0.15),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_base_datasets(config: dict):
    """Returns the raw ImageFolder datasets (without partitioning)."""
    img_size = EFFICIENTNET_INPUT_SIZES[config["model_version"]]
    train_tf, eval_tf = build_transforms(img_size)
    data_dir = Path(config["data_dir"])

    train_ds      = datasets.ImageFolder(data_dir / "train",      transform=train_tf)
    val_ds        = datasets.ImageFolder(data_dir / "validation",  transform=eval_tf)
    test_ds       = datasets.ImageFolder(data_dir / "test",        transform=eval_tf)
    train_eval_ds = datasets.ImageFolder(data_dir / "train",      transform=eval_tf)

    print(f"\n{'─'*60}")
    print(f"  Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"  Train      : {len(train_ds):>5} images")
    print(f"  Validation : {len(val_ds):>5} images")
    print(f"  Test       : {len(test_ds):>5} images")
    print(f"  Image size : {img_size}×{img_size}")
    print(f"{'─'*60}\n")

    return train_ds, val_ds, test_ds, train_eval_ds


def _loader_kwargs(config: dict):
    nw = config["num_workers"]
    return dict(
        num_workers=nw,
        pin_memory=config["pin_memory"],
        persistent_workers=(nw > 0 and config["persistent_workers"]),
        prefetch_factor=(config["prefetch_factor"] if nw > 0 else None),
    )


def build_global_loaders(val_ds, test_ds, train_eval_ds, config: dict):
    kw = _loader_kwargs(config)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                             shuffle=False, **kw)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=config["batch_size"],
                                   shuffle=False, **kw)
    return val_loader, test_loader, train_eval_loader


# ══════════════════════════════════════════════════════
#  DATASET PARTITIONING  (IID / Non-IID)
# ══════════════════════════════════════════════════════
def partition_dataset(train_ds, num_clients: int, iid: bool = True,
                      alpha: float = 0.5, seed: int = 42):
    """
    Splits training dataset indices into `num_clients` disjoint subsets.

    IID mode     : random equal split.
    Non-IID mode : Dirichlet(α) per class — lower α = more heterogeneous.

    Returns: list of index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(train_ds.targets)
    n_classes = len(train_ds.classes)
    client_indices = [[] for _ in range(num_clients)]

    if iid:
        all_idx = rng.permutation(len(targets))
        splits  = np.array_split(all_idx, num_clients)
        client_indices = [s.tolist() for s in splits]
    else:
        for c in range(n_classes):
            class_idx = np.where(targets == c)[0]
            rng.shuffle(class_idx)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            proportions = (proportions * len(class_idx)).astype(int)
            # Fix rounding so all samples are distributed
            proportions[-1] = len(class_idx) - proportions[:-1].sum()
            start = 0
            for k, p in enumerate(proportions):
                client_indices[k].extend(class_idx[start:start + p].tolist())
                start += p

    for k, idx in enumerate(client_indices):
        print(f"  Client {k+1}: {len(idx)} samples")
    return client_indices


def build_client_loader(train_ds, indices: list, config: dict):
    """Creates a weighted DataLoader for a single client's subset."""
    subset   = Subset(train_ds, indices)
    targets  = np.array(train_ds.targets)[indices]
    counts   = np.bincount(targets, minlength=len(train_ds.classes))
    counts   = np.where(counts == 0, 1, counts)   # avoid div-by-zero
    weights  = 1.0 / counts
    sample_w = [weights[t] for t in targets]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    kw = _loader_kwargs(config)
    return DataLoader(subset, batch_size=config["batch_size"],
                      sampler=sampler, drop_last=True, **kw)


# ══════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════
def build_model(version: str, num_classes: int, dropout_rate: float):
    """
    Pretrained EfficientNet with deeper classifier head.
    Backbone frozen; only head is trainable initially.
    """
    fn, weights = EFFICIENTNET_REGISTRY[version]
    model = fn(weights=weights)

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.GELU(),
        nn.Dropout(p=dropout_rate * 0.5),
        nn.Linear(1024, num_classes),
    )
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EfficientNet-{version.upper()} | head: {in_features}→1024→{num_classes}"
          f" | trainable: {trainable:,} params")
    return model


def unfreeze_last_n_blocks(model, n: int):
    base = model.module if isinstance(model, nn.DataParallel) else model
    total = len(base.features)
    for i, block in enumerate(base.features):
        for p in block.parameters():
            p.requires_grad = (i >= total - n)
    for p in base.classifier.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfroze last {n} backbone blocks | trainable: {trainable:,} params")


# ══════════════════════════════════════════════════════
#  FEDPROX LOSS  (CE + proximal regularisation)
# ══════════════════════════════════════════════════════
class FedProxLoss(nn.Module):
    """
    Computes: L_CE + (μ/2) * Σ ‖w_local - w_global‖²

    Args:
        base_criterion : any PyTorch loss (e.g. CrossEntropyLoss)
        global_params  : list of global model parameter tensors (detached)
        mu             : proximal coefficient
    """
    def __init__(self, base_criterion: nn.Module,
                 global_params: list, mu: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.global_params  = global_params   # already detached copies
        self.mu             = mu

    def forward(self, outputs, targets, local_model):
        ce_loss = self.base_criterion(outputs, targets)
        prox    = 0.0
        for local_p, global_p in zip(local_model.parameters(),
                                      self.global_params):
            prox += (local_p - global_p).norm(2) ** 2
        return ce_loss + (self.mu / 2.0) * prox


# ══════════════════════════════════════════════════════
#  MIXUP / CUTMIX  (CPU tensors)
# ══════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y):
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(x.size(0))
    _, _, H, W = x.shape
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_cut, y, y[idx], lam_adj


def mixed_loss(criterion_fn, pred, y_a, y_b, lam, local_model, global_params, mu):
    """Applies FedProx proximal term on top of mixed CE."""
    ce = lam * criterion_fn(pred, y_a) + (1 - lam) * criterion_fn(pred, y_b)
    prox = sum((lp - gp).norm(2) ** 2
               for lp, gp in zip(local_model.parameters(), global_params))
    return ce + (mu / 2.0) * prox


# ══════════════════════════════════════════════════════
#  LOCAL CLIENT UPDATE  (FedProx)
# ══════════════════════════════════════════════════════
def local_update(global_state: dict, client_loader: DataLoader,
                 num_classes: int, device: torch.device, config: dict,
                 round_num: int) -> tuple:
    """
    One round of local training for a single client.

    Args:
        global_state  : state_dict from the global model
        client_loader : DataLoader for this client's data partition
        num_classes   : number of flower-disease classes
        device        : torch device
        config        : FL_CONFIG dict
        round_num     : current global round (for unfreeze logic)

    Returns:
        (updated_state_dict, n_samples_trained)
    """
    # ── Build fresh local model, load global weights ─
    local_model = build_model(
        config["model_version"], num_classes, config["dropout_rate"])
    local_model.load_state_dict(global_state)
    local_model = local_model.to(device)

    # ── Unfreezing after unfreeze_round ──────────────
    if round_num >= config["unfreeze_round"]:
        unfreeze_last_n_blocks(local_model, config["unfreeze_last_n_blocks"])
        lr_factor = 0.05
    else:
        lr_factor = 1.0

    # ── Store global params for proximal term ────────
    global_params = [p.detach().clone() for p in local_model.parameters()]

    # ── Compute class weights from this client's data ─
    all_targets = []
    for _, lbls in client_loader:
        all_targets.extend(lbls.numpy())
    counts = np.bincount(np.array(all_targets), minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    cw = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    base_ce = nn.CrossEntropyLoss(weight=cw,
                                   label_smoothing=config["label_smoothing"])

    # ── Optimiser ─────────────────────────────────────
    lr = config["learning_rate"] * lr_factor
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, local_model.parameters()),
        lr=lr, weight_decay=config["weight_decay"],
    )

    # ── OneCycleLR ────────────────────────────────────
    total_steps = len(client_loader) * config["local_epochs"]
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps, pct_start=0.15,
        anneal_strategy="cos", div_factor=20, final_div_factor=1e3,
    )

    # ── Training loop ─────────────────────────────────
    local_model.train()
    n_samples = 0

    for epoch in range(config["local_epochs"]):
        running_loss = correct = total = 0

        for imgs, labels in client_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            # MixUp / CutMix
            apply_cutmix = np.random.rand() < config["cutmix_prob"]
            apply_mixup  = config["mixup_alpha"] > 0 and not apply_cutmix

            if apply_cutmix:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
            elif apply_mixup:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels,
                                                  config["mixup_alpha"])
            else:
                y_a, y_b, lam = labels, labels, 1.0

            optimizer.zero_grad(set_to_none=True)
            out  = local_model(imgs)
            loss = mixed_loss(base_ce, out, y_a, y_b, lam,
                              local_model, global_params, config["mu"])
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (lam * (preds == y_a).float() +
                        (1 - lam) * (preds == y_b).float()).sum().item()
            total   += labels.size(0)

        n_samples = total
        epoch_acc = correct / total if total > 0 else 0
        print(f"      local epoch {epoch+1}/{config['local_epochs']}"
              f"  loss: {running_loss/total:.4f}  acc: {epoch_acc:.4f}")

    return local_model.state_dict(), n_samples


# ══════════════════════════════════════════════════════
#  FEDERATED AVERAGING
# ══════════════════════════════════════════════════════
def federated_average(client_states: list, client_samples: list) -> dict:
    """
    Weighted FedAvg: Σ (n_i / n_total) * w_i

    Args:
        client_states  : list of state_dicts from each client
        client_samples : list of per-client sample counts

    Returns:
        Aggregated global state_dict.
    """
    total = sum(client_samples)
    global_state = copy.deepcopy(client_states[0])

    for key in global_state:
        global_state[key] = sum(
            (n / total) * s[key].float()
            for s, n in zip(client_states, client_samples)
        )

    return global_state


# ══════════════════════════════════════════════════════
#  EVALUATION UTILITIES
# ══════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        out    = model(imgs)
        loss   = criterion(out, labels)
        preds  = out.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


@torch.no_grad()
def evaluate_tta(model, loader, device, n_aug=5):
    """Test-Time Augmentation: averages n_aug forward passes with random flips."""
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        logits = torch.softmax(model(imgs), dim=1)
        for _ in range(n_aug - 1):
            aug    = torch.flip(imgs, dims=[3]) if np.random.rand() > 0.5 else imgs
            logits = logits + torch.softmax(model(aug), dim=1)
        preds = (logits / n_aug).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc, all_preds, all_labels


# ══════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ══════════════════════════════════════════════════════
def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, cmap, sub in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Blues", "YlOrRd"],
        ["(Counts)", "(Normalised)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, annot_kws={"size": 8},
                    **({} if fmt == "d" else {"vmin": 0, "vmax": 1}))
        ax.set_title(f"{title}\n{sub}", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_training_curves(history: dict, config: dict, save_path):
    rounds = range(1, len(history["val_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    style = dict(linewidth=2.2, marker="o", markersize=4)

    axes[0].plot(rounds, history["val_loss"],   label="Val Loss",
                 color="#F44336", **style)
    axes[0].set_title("Val Loss per Round",  fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.35)

    axes[1].plot(rounds, history["val_acc"],    label="Val Acc",
                 color="#4CAF50", **style)
    axes[1].set_title("Val Accuracy per Round", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    best_r   = int(np.argmax(history["val_acc"])) + 1
    best_acc = max(history["val_acc"])
    axes[1].annotate(
        f"Best: {best_acc:.4f}\n(round {best_r})",
        xy=(best_r, best_acc),
        xytext=(best_r + 1, best_acc - 0.07),
        arrowprops=dict(arrowstyle="->"), fontsize=10,
    )
    axes[1].legend(); axes[1].grid(alpha=0.35)

    ver = config["model_version"].upper()
    fig.suptitle(
        f"FedProx EfficientNet-{ver} — μ={config['mu']}"
        f" — {config['num_clients']} clients × {config['num_rounds']} rounds",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {save_path}")


# ══════════════════════════════════════════════════════
#  FULL EVALUATION  (Train + Val + Test)
# ══════════════════════════════════════════════════════
def evaluate_and_plot(model, history,
                      train_eval_loader, val_loader, test_loader,
                      class_names, criterion, device, config):
    results_dir = Path(config["results_dir"])
    version     = config["model_version"].upper()

    plot_training_curves(history, config,
                         results_dir / "training_curves_fedprox.png")

    print(f"\n{'─'*60}")
    splits = {
        "Train":      train_eval_loader,
        "Validation": val_loader,
        "Test":       test_loader,
    }
    results = {}
    for name, loader in splits.items():
        loss, acc, preds, labels = evaluate(model, loader, criterion, device)
        results[name] = (loss, acc, preds, labels)
        print(f"  {name:<14} | Loss: {loss:.4f}  Acc: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'─'*60}\n")

    # TTA on test
    if config.get("tta_n", 1) > 1:
        tta_acc, tta_preds, tta_labels = evaluate_tta(
            model, test_loader, device, n_aug=config["tta_n"])
        print(f"  Test Acc (TTA ×{config['tta_n']}): "
              f"{tta_acc:.4f}  ({tta_acc*100:.2f}%)")
        results["Test"] = (results["Test"][0], tta_acc, tta_preds, tta_labels)

    # Confusion matrices
    for name, (loss, acc, preds, labels) in results.items():
        plot_confusion_matrix(
            labels, preds, class_names,
            title=(f"{name} — FedProx EfficientNet-{version}"
                   f"\n(Acc: {acc*100:.2f}%)"),
            save_path=results_dir / f"confusion_matrix_{name.lower()}.png",
        )

    # Classification report
    _, _, preds_test, labels_test = results["Test"]
    report = classification_report(
        labels_test, preds_test, target_names=class_names, digits=4)
    print("\n  Classification Report (Test):\n")
    print(report)
    (results_dir / "classification_report_test.txt").write_text(report)

    # Save history
    with open(results_dir / "history_fedprox.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary table
    print(f"\n{'═'*60}")
    print(f"  {'Split':<14}  {'Loss':>8}  {'Accuracy':>12}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*12}")
    for name, (l, a, _, _) in results.items():
        print(f"  {name:<14}  {l:>8.4f}  {a:>9.4f}  ({a*100:.2f}%)")
    print(f"{'═'*60}\n")

    return results["Test"][1]


# ══════════════════════════════════════════════════════
#  FEDERATED TRAINING LOOP  (FedProx)
# ══════════════════════════════════════════════════════
def federated_train(config: dict) -> dict:
    """
    Main FedProx training loop.

    Round structure:
      1. Select fraction_fit × num_clients clients
      2. Each client runs local_update() (FedProx local loss)
      3. Aggregate with federated_average()
      4. Evaluate global model on validation set
      5. Checkpoint if best val_acc; early stop on patience

    Returns dict with best_val_acc, history, model, class_names, etc.
    """
    device = setup_device()
    set_seed(config["seed"])

    os.makedirs(config["save_dir"],    exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # ── Datasets ─────────────────────────────────────
    train_ds, val_ds, test_ds, train_eval_ds = build_base_datasets(config)
    num_classes  = len(train_ds.classes)
    class_names  = train_ds.classes

    val_loader, test_loader, train_eval_loader = build_global_loaders(
        val_ds, test_ds, train_eval_ds, config)

    # ── Partition training data across clients ────────
    print(f"\n  Partitioning train set → {config['num_clients']} clients "
          f"({'IID' if config['iid_partition'] else 'Non-IID'}) …")
    client_idx  = partition_dataset(
        train_ds,
        num_clients=config["num_clients"],
        iid=config["iid_partition"],
        alpha=config["dirichlet_alpha"],
        seed=config["seed"],
    )
    client_loaders = [
        build_client_loader(train_ds, idx, config) for idx in client_idx
    ]

    # ── Initialise global model ───────────────────────
    global_model = build_model(
        config["model_version"], num_classes, config["dropout_rate"])
    global_model = global_model.to(device)

    # Validation criterion (no proximal term for eval)
    targets      = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    class_weights = torch.tensor(1.0 / class_counts,
                                 dtype=torch.float32).to(device)
    val_criterion = nn.CrossEntropyLoss(weight=class_weights,
                                         label_smoothing=config["label_smoothing"])

    # ── History & early stopping ──────────────────────
    history = {"val_loss": [], "val_acc": []}
    best_val_acc     = 0.0
    best_state       = copy.deepcopy(global_model.state_dict())
    patience_counter = 0
    best_ckpt = Path(config["save_dir"]) / f"best_fedprox_{config['model_version']}.pth"

    n_select = max(1, int(config["num_clients"] * config["fraction_fit"]))

    print(f"\n{'═'*60}")
    print(f"  FedProx Federated Training")
    print(f"  Model     : EfficientNet-{config['model_version'].upper()}")
    print(f"  Clients   : {config['num_clients']}  (select {n_select}/round)")
    print(f"  Rounds    : {config['num_rounds']}")
    print(f"  Local ep  : {config['local_epochs']}")
    print(f"  μ (prox)  : {config['mu']}")
    print(f"  Device    : {device}")
    print(f"{'═'*60}\n")

    for rnd in range(1, config["num_rounds"] + 1):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  Round {rnd}/{config['num_rounds']}")

        # ── Progressive unfreezing notification ──────
        if rnd == config["unfreeze_round"]:
            print(f"  ► Unfreezing last {config['unfreeze_last_n_blocks']}"
                  f" backbone blocks from this round onward.")

        # ── Select clients ────────────────────────────
        selected = np.random.choice(
            config["num_clients"], n_select, replace=False).tolist()

        # ── Local updates ─────────────────────────────
        current_global_state = copy.deepcopy(global_model.state_dict())
        client_states  = []
        client_samples = []

        for k in selected:
            print(f"\n    ── Client {k+1} ──")
            state, n = local_update(
                current_global_state,
                client_loaders[k],
                num_classes,
                device,
                config,
                round_num=rnd,
            )
            client_states.append(state)
            client_samples.append(n)

        # ── Aggregate ─────────────────────────────────
        new_global_state = federated_average(client_states, client_samples)
        global_model.load_state_dict(new_global_state)

        # ── Validate ──────────────────────────────────
        vl_loss, vl_acc, _, _ = evaluate(
            global_model, val_loader, val_criterion, device)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"\n  Round {rnd:02d} | val_loss: {vl_loss:.4f}"
              f" | val_acc: {vl_acc:.4f} ({vl_acc*100:.2f}%)"
              f" | {elapsed:.1f}s")

        # ── Checkpoint ────────────────────────────────
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(global_model.state_dict())
            torch.save({
                "round":       rnd,
                "model_state": best_state,
                "val_acc":     best_val_acc,
                "class_names": class_names,
                "config":      config,
            }, best_ckpt)
            patience_counter = 0
            print(f"  ✓ New best val acc: {best_val_acc:.4f}  → {best_ckpt}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at round {rnd}.")
                break

        gc.collect()

    # Load best global model
    global_model.load_state_dict(best_state)
    print(f"\n  Best validation accuracy (FL): {best_val_acc:.4f}")

    return {
        "model":          global_model,
        "history":        history,
        "best_val_acc":   best_val_acc,
        "class_names":    class_names,
        "val_criterion":  val_criterion,
        "train_eval_loader": train_eval_loader,
        "val_loader":     val_loader,
        "test_loader":    test_loader,
        "device":         device,
    }


# ══════════════════════════════════════════════════════
#  SINGLE-IMAGE INFERENCE
# ══════════════════════════════════════════════════════
def predict_single(model, image_path: str, class_names: list,
                   model_version: str, device: torch.device):
    img_size   = EFFICIENTNET_INPUT_SIZES[model_version]
    _, eval_tf = build_transforms(img_size)
    img    = Image.open(image_path).convert("RGB")
    tensor = eval_tf(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    top3 = np.argsort(probs)[::-1][:3]
    print(f"\n  Top-3 predictions for '{image_path}':")
    for i, idx in enumerate(top3, 1):
        print(f"    {i}. {class_names[idx]:<45} {probs[idx]*100:.2f}%")
    return class_names[top3[0]], probs[top3[0]]


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    results = federated_train(FL_CONFIG)

    test_acc = evaluate_and_plot(
        model             = results["model"],
        history           = results["history"],
        train_eval_loader = results["train_eval_loader"],
        val_loader        = results["val_loader"],
        test_loader       = results["test_loader"],
        class_names       = results["class_names"],
        criterion         = results["val_criterion"],
        device            = results["device"],
        config            = FL_CONFIG,
    )

    print(f"\n{'═'*60}")
    print(f"  FedProx Training Complete")
    print(f"  Best Val Acc : {results['best_val_acc']*100:.2f}%")
    print(f"  Final Test   : {test_acc*100:.2f}%")
    print(f"  Checkpoint   : ./checkpoints/best_fedprox_{FL_CONFIG['model_version']}.pth")
    print(f"{'═'*60}\n")
