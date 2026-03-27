"""
Flower Disease Classification — EfficientNet (GPU-Optimised)
════════════════════════════════════════════════════════════════
Dataset : dataset_flat_structure/  (train / validation / test)
Classes : Chrysanthemum_Bacterial_Leaf_Spot, Chrysanthemum_Healthy,
          Chrysanthemum_Septoria_Leaf_Spot,
          Jasmine_Healthy, Jasmine_Multiple, Jasmine_Rust

GPU Optimisations added
──────────────────────
 • Auto GPU detection + device info print (name, VRAM, CUDA version)
 • cudnn.benchmark = True       → auto-selects fastest conv algorithm
 • AMP (torch.cuda.amp)         → FP16 forward/backward (2× speed, 2× VRAM savings)
 • GradScaler                   → stable FP16 training
 • non_blocking=True transfers  → overlaps CPU→GPU copy with compute
 • pin_memory=True              → faster host→device pinned memory transfer
 • persistent_workers=True      → keeps DataLoader workers alive across epochs
 • prefetch_factor=4            → pre-loads batches in background
 • Multi-GPU support (nn.DataParallel) → auto-splits batch across all GPUs
 • torch.compile() (PyTorch 2+) → graph-level kernel fusion for extra speed
 • GPU memory management        → clears cache before training
 • Per-epoch GPU memory monitor  → shows peak VRAM usage
"""

import os
import gc
import time
import copy
import json
import warnings
import numpy as np
from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
#  GPU SETUP — called before anything else
# ══════════════════════════════════════════════════════
def setup_gpu():
    """
    Detects GPU(s), prints hardware info, sets all CUDA flags for
    maximum throughput, and returns the primary device.
    Falls back to CPU with a clear warning if no GPU is found.
    """
    if not torch.cuda.is_available():
        print("\n" + "⚠" * 55)
        print("  WARNING: No CUDA GPU detected — running on CPU.")
        print("  Training will be significantly slower.")
        print("⚠" * 55 + "\n")
        return torch.device("cpu"), 1

    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")

    print("\n" + "═" * 60)
    print("  GPU INFORMATION")
    print("═" * 60)
    print(f"  CUDA version      : {torch.version.cuda}")
    print(f"  PyTorch version   : {torch.__version__}")
    print(f"  Number of GPUs    : {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / (1024 ** 3)
        print(f"  GPU {i}  : {props.name}")
        print(f"          VRAM      : {vram:.1f} GB")
        print(f"          Compute   : {props.major}.{props.minor}")
        print(f"          SM count  : {props.multi_processor_count}")
    print("═" * 60 + "\n")

    # ── CUDA performance flags ──────────────────
    torch.backends.cudnn.enabled    = True
    torch.backends.cudnn.benchmark  = True   # fastest conv algorithm auto-select
    torch.backends.cudnn.deterministic = False  # deterministic=False for speed

    # Clear any leftover allocations
    torch.cuda.empty_cache()
    gc.collect()

    return device, n_gpus


def gpu_memory_stats(device):
    """Returns current and peak GPU memory in MB."""
    if device.type != "cuda":
        return 0.0, 0.0
    cur  = torch.cuda.memory_allocated(device)  / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return cur, peak


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ══════════════════════════════════════════════════════
#  HELPER — parse env values
# ══════════════════════════════════════════════════════
def _env(key, default):
    """Read an env var and cast to the same type as *default*."""
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
#  CONFIG  (reads from .env → falls back to defaults)
# ══════════════════════════════════════════════════════
CONFIG = {
    # ── Paths ──────────────────────────────────────
    "data_dir":    _env("DATA_DIR",    "./dataset_flat_structure"),
    "save_dir":    _env("SAVE_DIR",    "./checkpoints"),
    "results_dir": _env("RESULTS_DIR", "./results"),

    # ── Model ──────────────────────────────────────
    "model_version": _env("MODEL_VERSION", "b3"),

    # ── Training ───────────────────────────────────
    "num_epochs":      _env("NUM_EPOCHS",      20),
    "batch_size":      _env("BATCH_SIZE",      16),
    "learning_rate":   _env("LEARNING_RATE",   3e-4),
    "weight_decay":    _env("WEIGHT_DECAY",    1e-4),
    "label_smoothing": _env("LABEL_SMOOTHING", 0.1),
    "dropout_rate":    _env("DROPOUT_RATE",    0.35),

    # ── Augmentation ───────────────────────────────
    "mixup_alpha": _env("MIXUP_ALPHA", 0.3),
    "cutmix_prob": _env("CUTMIX_PROB", 0.4),

    # ── Transfer learning ──────────────────────────
    "unfreeze_epoch":         _env("UNFREEZE_EPOCH",         8),
    "unfreeze_last_n_blocks": _env("UNFREEZE_LAST_N_BLOCKS", 5),

    # ── Early stopping ─────────────────────────────
    "patience": _env("PATIENCE", 12),

    # ── DataLoader GPU flags ───────────────────────
    "num_workers":       _env("NUM_WORKERS",       4),
    "pin_memory":        _env("PIN_MEMORY",        True),
    "prefetch_factor":   _env("PREFETCH_FACTOR",   4),
    "persistent_workers": _env("PERSISTENT_WORKERS", True),

    # ── AMP (Automatic Mixed Precision) ────────────
    "use_amp": _env("USE_AMP", True),

    # ── Multi-GPU ──────────────────────────────────
    "use_multi_gpu": _env("USE_MULTI_GPU", True),

    # ── torch.compile (PyTorch 2.0+) ───────────────
    "compile_model": _env("COMPILE_MODEL", False),

    # ── TTA ────────────────────────────────────────
    "tta_n": _env("TTA_N", 5),

    # ── Misc ───────────────────────────────────────
    "seed": _env("SEED", 42),
}

EFFICIENTNET_INPUT_SIZES = {
    "b0": 224, "b1": 240, "b2": 260, "b3": 300,
    "b4": 380, "b5": 456, "b6": 528, "b7": 600,
}

EFFICIENTNET_REGISTRY = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    "b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "b5": (efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1),
    "b6": (efficientnet_b6, EfficientNet_B6_Weights.IMAGENET1K_V1),
    "b7": (efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1),
}


# ══════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # NOTE: deterministic=False kept intentionally for GPU speed


# ══════════════════════════════════════════════════════
#  CUSTOM PREPROCESSING TRANSFORMS
# ══════════════════════════════════════════════════════
class CLAHE:
    """
    Contrast Limited Adaptive Histogram Equalisation in LAB space.
    Sharpens disease lesion boundaries without altering hue.
    Requires: pip install opencv-python-headless
    """
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
            return img   # silent fallback


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
        # ── Custom preprocessing ──
        CLAHE(clip_limit=2.5, tile=(8, 8), p=0.5),
        SharpenFilter(radius=1.5, percent=130, threshold=2, p=0.4),
        RandomBrightnessContrast(br=(0.70, 1.30), cr=(0.70, 1.30), p=0.5),
        # ── Colour ──
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.12),
        transforms.RandomGrayscale(p=0.05),
        # ── Geometry ──
        transforms.RandomPerspective(distortion_scale=0.35, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.85, 1.15), shear=10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        RandomGridShuffle(grid=3, p=0.15),
        # ── AutoAugment ──
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        # ── Tensor ──
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


def make_weighted_sampler(dataset):
    counts  = np.bincount(dataset.targets)
    weights = 1.0 / counts
    sample_w = [weights[t] for t in dataset.targets]
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def build_dataloaders(config: dict):
    img_size = EFFICIENTNET_INPUT_SIZES[config["model_version"]]
    train_tf, eval_tf = build_transforms(img_size)
    data_dir = Path(config["data_dir"])

    train_ds      = datasets.ImageFolder(data_dir / "train",      transform=train_tf)
    val_ds        = datasets.ImageFolder(data_dir / "validation",  transform=eval_tf)
    test_ds       = datasets.ImageFolder(data_dir / "test",        transform=eval_tf)
    train_eval_ds = datasets.ImageFolder(data_dir / "train",      transform=eval_tf)

    sampler = make_weighted_sampler(train_ds)

    # ── GPU-optimised DataLoader kwargs ─────────────────
    # persistent_workers keeps workers alive → no fork overhead per epoch
    # pin_memory pages memory → DMA transfer to GPU is faster
    # prefetch_factor loads N batches ahead in background
    nw = config["num_workers"]
    loader_kw = dict(
        num_workers=nw,
        pin_memory=config["pin_memory"],
        persistent_workers=(nw > 0 and config["persistent_workers"]),
        prefetch_factor=(config["prefetch_factor"] if nw > 0 else None),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        sampler=sampler, drop_last=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, **loader_kw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"],
        shuffle=False, **loader_kw,
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=config["batch_size"],
        shuffle=False, **loader_kw,
    )

    print(f"\n{'─'*60}")
    print(f"  Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"  Train      : {len(train_ds):>5} images")
    print(f"  Validation : {len(val_ds):>5} images")
    print(f"  Test       : {len(test_ds):>5} images")
    print(f"  Image size : {img_size}×{img_size}")
    print(f"  Workers    : {nw}  |  pin_memory: {config['pin_memory']}"
          f"  |  prefetch: {config['prefetch_factor']}")
    print(f"{'─'*60}\n")

    return (train_loader, val_loader, test_loader,
            train_eval_loader, train_ds.classes)


# ══════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════
def build_model(version: str, num_classes: int, dropout_rate: float,
                n_gpus: int, config: dict):
    """
    Pretrained EfficientNet with a deeper GPU-resident classifier head.
    Wraps in nn.DataParallel when multiple GPUs are available.
    Optionally compiles with torch.compile (PyTorch 2+).
    """
    fn, weights = EFFICIENTNET_REGISTRY[version]
    model = fn(weights=weights)

    # Freeze backbone
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

    # ── Multi-GPU: DataParallel ──────────────────────
    if config["use_multi_gpu"] and n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  Using nn.DataParallel across {n_gpus} GPUs")

    # ── torch.compile (PyTorch 2+) ──────────────────
    if config.get("compile_model", False):
        try:
            model = torch.compile(model)
            print("  torch.compile() applied — kernel fusion enabled")
        except Exception as e:
            print(f"  torch.compile() skipped: {e}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EfficientNet-{version.upper()} ready  |  "
          f"head: {in_features}→1024→{num_classes}  |  "
          f"trainable: {trainable:,} params")
    return model


def unfreeze_last_n_blocks(model, n: int):
    # Unwrap DataParallel if needed
    base = model.module if isinstance(model, nn.DataParallel) else model
    total = len(base.features)
    for i, block in enumerate(base.features):
        for p in block.parameters():
            p.requires_grad = (i >= total - n)
    for p in base.classifier.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfroze last {n} blocks  |  Trainable params: {trainable:,}")


# ══════════════════════════════════════════════════════
#  MIXUP / CUTMIX  (GPU-resident — no CPU round-trip)
# ══════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y):
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(x.size(0), device=x.device)
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


def mixed_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════
#  TRAINING LOOP  (AMP + non-blocking transfers)
# ══════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, scheduler, device, config):
    model.train()
    running_loss = correct = total = 0
    n_batches = len(loader)

    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        # non_blocking=True overlaps the CPU→GPU copy with GPU compute
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── MixUp / CutMix (on GPU) ───────────
        apply_cutmix = np.random.rand() < config["cutmix_prob"]
        apply_mixup  = config["mixup_alpha"] > 0 and not apply_cutmix

        if apply_cutmix:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
        elif apply_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, config["mixup_alpha"])
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        # ── AMP forward pass ──────────────────
        with torch.autocast(device_type=device.type,
                            enabled=config["use_amp"]):
            out  = model(imgs)
            loss = mixed_loss(criterion, out, y_a, y_b, lam)

        # ── AMP backward ──────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        preds    = out.argmax(dim=1)
        correct += (lam * (preds == y_a).float() +
                    (1 - lam) * (preds == y_b).float()).sum().item()
        total   += labels.size(0)

        # ── Progress printing ─────────────────
        if batch_idx % 50 == 0 or batch_idx == n_batches:
            avg_loss = running_loss / total
            avg_acc  = correct / total
            print(f"    Batch {batch_idx:>4d}/{n_batches}"
                  f"  | loss: {avg_loss:.4f}  acc: {avg_acc:.4f}", flush=True)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    running_loss = correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)

        preds = out.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ══════════════════════════════════════════════════════
#  TEST-TIME AUGMENTATION (TTA)
# ══════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_tta(model, loader, device, n_aug=5, use_amp=True):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = torch.softmax(model(imgs), dim=1)
            for _ in range(n_aug - 1):
                aug = torch.flip(imgs, dims=[3]) if np.random.rand() > 0.5 else imgs
                logits = logits + torch.softmax(model(aug), dim=1)

        preds = (logits / n_aug).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc, all_preds, all_labels


# ══════════════════════════════════════════════════════
#  MAIN TRAINER
# ══════════════════════════════════════════════════════
def train(config: dict):
    # ── GPU setup ─────────────────────────────────────
    device, n_gpus = setup_gpu()
    set_seed(config["seed"])

    os.makedirs(config["save_dir"],    exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # ── Data ──────────────────────────────────────────
    (train_loader, val_loader, test_loader,
     train_eval_loader, class_names) = build_dataloaders(config)
    num_classes = len(class_names)

    # ── Model → GPU ───────────────────────────────────
    model = build_model(config["model_version"], num_classes,
                        config["dropout_rate"], n_gpus, config)
    model = model.to(device)

    # ── Loss ──────────────────────────────────────────
    targets      = np.array(train_loader.dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = torch.tensor(1.0 / class_counts,
                                 dtype=torch.float32).to(device,
                                                          non_blocking=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                     label_smoothing=config["label_smoothing"])

    # ── Optimiser ─────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fused=True if device.type == "cuda" else False,  # fused AdamW on GPU
    )

    # ── OneCycleLR ────────────────────────────────────
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = OneCycleLR(
        optimizer, max_lr=config["learning_rate"],
        total_steps=total_steps, pct_start=0.1,
        anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
    )

    # ── AMP GradScaler ────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(
        enabled=(config["use_amp"] and device.type == "cuda")
    )

    # ── History & early stopping ──────────────────────
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
    best_val_acc     = 0.0
    best_wts         = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_ckpt = (Path(config["save_dir"]) /
                 f"best_efficientnet_{config['model_version']}.pth")

    print(f"\n{'═'*60}")
    print(f"  Training EfficientNet-{config['model_version'].upper()}"
          f"  |  {num_classes} classes  |  {config['num_epochs']} epochs")
    print(f"  AMP: {config['use_amp']}  |  "
          f"Multi-GPU: {n_gpus > 1 and config['use_multi_gpu']}")
    print(f"{'═'*60}\n")

    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()
        reset_peak_memory(device)

        # ── Progressive unfreezing ────────────────
        if epoch == config["unfreeze_epoch"]:
            print(f"\n  [Epoch {epoch}] Unfreezing last "
                  f"{config['unfreeze_last_n_blocks']} blocks …")
            unfreeze_last_n_blocks(model, config["unfreeze_last_n_blocks"])
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config["learning_rate"] * 0.05,
                weight_decay=config["weight_decay"],
                fused=True if device.type == "cuda" else False,
            )
            remaining = (config["num_epochs"] - epoch + 1) * len(train_loader)
            scheduler = OneCycleLR(
                optimizer, max_lr=config["learning_rate"] * 0.05,
                total_steps=remaining, pct_start=0.05,
                anneal_strategy="cos", div_factor=10, final_div_factor=1e3,
            )

        # ── Train / Val ───────────────────────────
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, device, config,
        )
        vl_loss, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, device, config["use_amp"]
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        _, peak_mb = gpu_memory_stats(device)
        print(f"  Epoch {epoch:03d}/{config['num_epochs']}"
              f"  | train {tr_loss:.4f}/{tr_acc:.4f}"
              f"  | val {vl_loss:.4f}/{vl_acc:.4f}"
              f"  | {elapsed:.1f}s"
              f"  | VRAM peak {peak_mb:.0f} MB")

        # ── Checkpoint ────────────────────────────
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_wts     = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch":       epoch,
                "model_state": best_wts,
                "val_acc":     best_val_acc,
                "class_names": class_names,
                "config":      config,
            }, best_ckpt)
            patience_counter = 0
            print(f"  ✓ Best val acc: {best_val_acc:.4f}  → {best_ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

        # Free unused GPU memory each epoch
        torch.cuda.empty_cache()

    model.load_state_dict(best_wts)
    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
    return (model, history,
            train_eval_loader, val_loader, test_loader,
            class_names, criterion, device)


# ══════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ══════════════════════════════════════════════════════
def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, cmap, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Blues", "YlOrRd"],
        ["(Counts)", "(Normalised)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, annot_kws={"size": 9},
                    **({} if fmt == "d" else {"vmin": 0, "vmax": 1}))
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_training_curves(history, config, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    style = dict(linewidth=2.2, marker="o", markersize=3)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss",
                 color="#2196F3", **style)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",
                 color="#F44336", **style)
    axes[0].set_title("Loss per Epoch",     fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.35)
    axes[0].fill_between(epochs, history["train_loss"],
                         history["val_loss"], alpha=0.07, color="gray")

    axes[1].plot(epochs, history["train_acc"], label="Train Acc",
                 color="#4CAF50", **style)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",
                 color="#FF9800", **style)
    axes[1].set_title("Accuracy per Epoch", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(alpha=0.35)

    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc   = max(history["val_acc"])
    axes[1].annotate(
        f"Best: {best_acc:.4f}\n(epoch {best_epoch})",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 1, best_acc - 0.07),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

    fig.suptitle(
        f"EfficientNet-{config['model_version'].upper()} — Training Curves",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {save_path}")


# ══════════════════════════════════════════════════════
#  FULL EVALUATION (Train + Val + Test confusion matrices)
# ══════════════════════════════════════════════════════
def evaluate_and_plot(model, history,
                      train_eval_loader, val_loader, test_loader,
                      class_names, criterion, device, config):
    results_dir = Path(config["results_dir"])
    version     = config["model_version"].upper()
    use_amp     = config["use_amp"]

    # ── 1. Training curves ──────────────────────────
    plot_training_curves(history, config,
                         results_dir / "training_curves.png")

    # ── 2. Evaluate all three splits ────────────────
    print(f"\n{'─'*60}")
    splits = {
        "Train":      train_eval_loader,
        "Validation": val_loader,
        "Test":       test_loader,
    }
    results = {}
    for name, loader in splits.items():
        loss, acc, preds, labels = evaluate(
            model, loader, criterion, device, use_amp)
        results[name] = (loss, acc, preds, labels)
        print(f"  {name:<14} | Loss: {loss:.4f}  Acc: {acc:.4f}"
              f"  ({acc*100:.2f}%)")
    print(f"{'─'*60}\n")

    # ── 3. TTA on test set ───────────────────────────
    if config.get("tta_n", 1) > 1:
        tta_acc, tta_preds, tta_labels = evaluate_tta(
            model, test_loader, device,
            n_aug=config["tta_n"], use_amp=use_amp,
        )
        print(f"  Test Acc (TTA ×{config['tta_n']}): "
              f"{tta_acc:.4f}  ({tta_acc*100:.2f}%)")
        results["Test"] = (results["Test"][0], tta_acc, tta_preds, tta_labels)

    # ── 4. Confusion matrices for all splits ────────
    for name, (loss, acc, preds, labels) in results.items():
        plot_confusion_matrix(
            labels, preds, class_names,
            title=(f"{name} Confusion Matrix — EfficientNet-{version}"
                   f"\n(Acc: {acc*100:.2f}%)"),
            save_path=results_dir / f"confusion_matrix_{name.lower()}.png",
        )

    # ── 5. Classification report (test) ─────────────
    _, _, preds_test, labels_test = results["Test"]
    report = classification_report(
        labels_test, preds_test, target_names=class_names, digits=4)
    print("\n  Classification Report (Test):\n")
    print(report)
    (results_dir / "classification_report_test.txt").write_text(report)

    # ── 6. Save history ──────────────────────────────
    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── 7. Summary table ─────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {'Split':<14}  {'Loss':>8}  {'Accuracy':>12}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*12}")
    for name, (l, a, _, _) in results.items():
        print(f"  {name:<14}  {l:>8.4f}  {a:>9.4f}  ({a*100:.2f}%)")
    print(f"{'═'*60}\n")

    return results["Test"][1]


# ══════════════════════════════════════════════════════
#  SINGLE-IMAGE INFERENCE
# ══════════════════════════════════════════════════════
def predict_single(model, image_path: str, class_names: list,
                   model_version: str, device):
    img_size   = EFFICIENTNET_INPUT_SIZES[model_version]
    _, eval_tf = build_transforms(img_size)
    img    = Image.open(image_path).convert("RGB")
    tensor = eval_tf(img).unsqueeze(0).to(device, non_blocking=True)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type):
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
    (model, history,
     train_eval_loader, val_loader, test_loader,
     class_names, criterion, device) = train(CONFIG)

    test_acc = evaluate_and_plot(
        model, history,
        train_eval_loader, val_loader, test_loader,
        class_names, criterion, device, CONFIG,
    )

    print(f"  Final test accuracy : {test_acc*100:.2f}%")
    print("  Done.\n")
