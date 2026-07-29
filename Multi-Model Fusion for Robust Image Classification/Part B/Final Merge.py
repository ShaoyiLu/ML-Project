import os
import copy
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms


def set_seed(seed=771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(771)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"Task2_data"
SAVE_DIR = r"checkpoints_taskB"
os.makedirs(SAVE_DIR, exist_ok=True)

CACHE_PATH = os.path.join(SAVE_DIR, "final_merge_feature_cache.pt")
BEST_HEAD_PATH = os.path.join(SAVE_DIR, "final_merged_fusion_head_best.pth")
USE_MODEL3_HARD3 = False
USE_MODEL3_173_RESCUE = True
MODEL3_173_RESCUE_THRESHOLD = 0.45
MERGE_CONFIG_VERSION = "source_routed_v7_m1patch03_m2comp10_173rescue045"

TRAIN_PATHS = [
    os.path.join(ROOT, "train_dataB_model_1.pth"),
    os.path.join(ROOT, "train_dataB_model_2.pth"),
    os.path.join(ROOT, "train_dataB_model_3.pth"),
]

VAL_PATHS = [
    os.path.join(ROOT, "val_dataB_model_1.pth"),
    os.path.join(ROOT, "val_dataB_model_2.pth"),
    os.path.join(ROOT, "val_dataB_model_3.pth"),
]

GLOBAL_LABELS = [24, 34, 80, 124, 125, 130, 135, 137, 159, 173, 201, 202]
GLOBAL_MAP = {lab: i for i, lab in enumerate(GLOBAL_LABELS)}

MODEL1_LABELS = [34, 137, 159, 173, 201]
MODEL2_LABELS = [24, 34, 80, 135, 202]
MODEL3_LABELS = [124, 125, 130, 173, 202]
MODEL3_HARD3_LABELS = [125, 130, 173]

MEAN = [0.4920, 0.4653, 0.3957]
STD = [0.2401, 0.2301, 0.2362]
normalize_tf = transforms.Normalize(MEAN, STD)


class TaskBUnionDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.samples = []
        self.transform = transform

        for source_id, path in enumerate(paths, start=1):
            raw = torch.load(path, map_location="cpu")
            data = raw["data"]
            labels = raw["labels"]

            for i in range(len(labels)):
                raw_label = int(labels[i])
                self.samples.append(
                    {
                        "image": data[i],
                        "raw_label": raw_label,
                        "global_label": GLOBAL_MAP[raw_label],
                        "source_id": source_id,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        x = item["image"].float() / 255.0

        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)

        if self.transform is not None:
            x = self.transform(x)

        return (
            x,
            torch.tensor(item["global_label"], dtype=torch.long),
            torch.tensor(item["raw_label"], dtype=torch.long),
            torch.tensor(item["source_id"], dtype=torch.long),
        )


class ExpertCNNBN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),

            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),

            nn.Conv2d(384, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.35),

            nn.Linear(384, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(self.dense(self.features(x)))


class ExpertCNNGN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 192, 3, padding=1),
            nn.GroupNorm(8, 192),
            nn.SiLU(inplace=True),

            nn.Conv2d(192, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.GroupNorm(16, 384),
            nn.SiLU(inplace=True),

            nn.Conv2d(384, 512, 3, padding=1),
            nn.GroupNorm(16, 512),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.38),

            nn.Linear(384, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.28),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(self.dense(self.features(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0, pool=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        ]

        if pool:
            layers.append(nn.MaxPool2d(2))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CompactExpertCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64, groups=8, dropout=0.05, pool=True),
            ConvBlock(64, 128, groups=8, dropout=0.10, pool=True),
            ConvBlock(128, 192, groups=8, dropout=0.15, pool=True),
            ConvBlock(192, 256, groups=16, dropout=0.20, pool=False),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FusionHead(nn.Module):
    def __init__(self, in_dim=30, num_classes=12):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.18),
            nn.Linear(96, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model(model, path):
    if not os.path.exists(path):
        print("Skip missing checkpoint:", path)
        return None

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print("Loaded:", path)
    return model


def build_expert_models():
    models = []

    specs = [
        ("m1_old", ExpertCNNBN(), os.path.join(SAVE_DIR, "modelB_1_expert_best.pth")),
        ("m1_lowlr", ExpertCNNBN(), os.path.join(SAVE_DIR, "modelB_1_expert_best_lowlr.pth")),
        ("m1_gn", ExpertCNNGN(), os.path.join(SAVE_DIR, "modelB_1_expert_best_groupnorm.pth")),
        ("m1_137201", ExpertCNNBN(), os.path.join(SAVE_DIR, "modelB_1_expert_best_137201.pth")),
        ("m2_lowlr", ExpertCNNBN(), os.path.join(SAVE_DIR, "modelB_2_expert_best_lowlr.pth")),
        ("m2_comp", ExpertCNNBN(), os.path.join(SAVE_DIR, "modelB_2_expert_best_complement.pth")),
        ("m3_lowlr", ExpertCNNBN(), os.path.join(SAVE_DIR, "modelB_3_expert_best_lowlr.pth")),
        ("m3_transfer", ExpertCNNBN(), os.path.join(SAVE_DIR, "modelB_3_expert_best_transfer.pth")),
        ("m3_rebuild", CompactExpertCNN(), os.path.join(SAVE_DIR, "modelB_3_expert_best_rebuild.pth")),
    ]

    if USE_MODEL3_HARD3:
        specs.append(
            ("m3_hard3", CompactExpertCNN(num_classes=3), os.path.join(SAVE_DIR, "modelB_3_hard3_best.pth"))
        )

    if USE_MODEL3_173_RESCUE:
        specs.append(
            ("m3_173_rescue", ExpertCNNBN(num_classes=2), os.path.join(SAVE_DIR, "modelB_3_173_rescue_best.pth"))
        )

    for name, model, path in specs:
        loaded = load_model(model, path)
        if loaded is not None:
            models.append((name, loaded))

    return models


def merge_local_outputs(outputs):
    m1_logits = []
    m2_logits = []
    m3_logits = []

    if "m1_old" in outputs:
        m1_logits.append(outputs["m1_old"] * 1.0)
    if "m1_lowlr" in outputs:
        m1_logits.append(outputs["m1_lowlr"] * 1.0)
    if "m1_gn" in outputs:
        m1_logits.append(outputs["m1_gn"] * 0.6)
    if "m1_137201" in outputs:
        m1_logits.append(outputs["m1_137201"] * 0.3)

    if "m2_lowlr" in outputs:
        m2_logits.append(outputs["m2_lowlr"] * 1.0)
    if "m2_comp" in outputs:
        m2_logits.append(outputs["m2_comp"] * 1.0)

    if "m3_lowlr" in outputs:
        m3_logits.append(outputs["m3_lowlr"] * 0.6)
    if "m3_transfer" in outputs:
        m3_logits.append(outputs["m3_transfer"] * 0.6)
    if "m3_rebuild" in outputs:
        m3_logits.append(outputs["m3_rebuild"] * 0.8)

    m1 = sum(m1_logits) if m1_logits else None
    m2 = sum(m2_logits) if m2_logits else None
    m3 = sum(m3_logits) if m3_logits else None

    hard3 = outputs.get("m3_hard3")
    rescue173 = outputs.get("m3_173_rescue")

    return m1, m2, m3, hard3, rescue173


def expert_features_from_logits(m1, m2, m3, hard3, rescue173):
    parts = []

    for logits in [m1, m2, m3]:
        if logits is None:
            parts.append(torch.zeros(1, 5))
            parts.append(torch.zeros(1, 5))
        else:
            parts.append(logits.cpu())
            parts.append(torch.softmax(logits, dim=1).cpu())

    if hard3 is not None:
        parts.append(hard3.cpu())
        parts.append(torch.softmax(hard3, dim=1).cpu())

    if rescue173 is not None:
        parts.append(rescue173.cpu())
        parts.append(torch.softmax(rescue173, dim=1).cpu())

    return torch.cat(parts, dim=1)


def precompute_features(models, dataset, cache_name):
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    feature_chunks = []
    label_chunks = []
    raw_label_chunks = []
    source_chunks = []

    with torch.no_grad():
        for x, y, raw_y, source_id in loader:
            x = x.to(device)

            outputs = {}
            for name, model in models:
                outputs[name] = model(x)

            m1, m2, m3, hard3, rescue173 = merge_local_outputs(outputs)
            feats = expert_features_from_logits(m1, m2, m3, hard3, rescue173)

            feature_chunks.append(feats)
            label_chunks.append(y.cpu())
            raw_label_chunks.append(raw_y.cpu())
            source_chunks.append(source_id.cpu())

    features = torch.cat(feature_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    raw_labels = torch.cat(raw_label_chunks, dim=0)
    sources = torch.cat(source_chunks, dim=0)

    print(f"{cache_name}: features={tuple(features.shape)}, labels={tuple(labels.shape)}")
    return features, labels, raw_labels, sources


def checkpoint_signature():
    names = [
        "modelB_1_expert_best.pth",
        "modelB_1_expert_best_lowlr.pth",
        "modelB_1_expert_best_groupnorm.pth",
        "modelB_1_expert_best_137201.pth",
        "modelB_2_expert_best_lowlr.pth",
        "modelB_2_expert_best_complement.pth",
        "modelB_3_expert_best_lowlr.pth",
        "modelB_3_expert_best_transfer.pth",
        "modelB_3_expert_best_rebuild.pth",
    ]

    sig = {}
    sig["merge_config_version"] = MERGE_CONFIG_VERSION
    if USE_MODEL3_HARD3:
        names.append("modelB_3_hard3_best.pth")
    if USE_MODEL3_173_RESCUE:
        names.append("modelB_3_173_rescue_best.pth")
    for name in names:
        path = os.path.join(SAVE_DIR, name)
        if os.path.exists(path):
            sig[name] = os.path.getmtime(path)

    return sig


def load_or_build_cache():
    current_signature = checkpoint_signature()

    if os.path.exists(CACHE_PATH):
        print("Loading cached expert features:", CACHE_PATH)
        cache = torch.load(CACHE_PATH, map_location="cpu")

        if cache.get("checkpoint_signature") == current_signature:
            return cache

        print("Checkpoint set changed. Rebuilding expert feature cache.")

    print("Building expert feature cache. First run can be slow on CPU.")
    models = build_expert_models()

    val_dataset = TaskBUnionDataset(VAL_PATHS, transform=normalize_tf)

    val_features, val_labels, val_raw, val_sources = precompute_features(
        models,
        val_dataset,
        "val",
    )

    cache = {
        "train_features": torch.empty(0),
        "train_labels": torch.empty(0, dtype=torch.long),
        "train_raw": torch.empty(0, dtype=torch.long),
        "train_sources": torch.empty(0, dtype=torch.long),
        "val_features": val_features,
        "val_labels": val_labels,
        "val_raw": val_raw,
        "val_sources": val_sources,
        "global_labels": GLOBAL_LABELS,
        "checkpoint_signature": current_signature,
    }

    torch.save(cache, CACHE_PATH)
    print("Saved cache:", CACHE_PATH)
    return cache


def make_class_weights(labels):
    counts = Counter(int(x) for x in labels)
    weights = []

    for i in range(len(GLOBAL_LABELS)):
        weights.append(1.0 / counts[i])

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


def train_fusion_head(cache):
    x_train = cache["train_features"]
    y_train = cache["train_labels"]
    x_val = cache["val_features"]
    y_val = cache["val_labels"]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )

    model = FusionHead(in_dim=x_train.size(1), num_classes=len(GLOBAL_LABELS)).to(device)

    class_weights = make_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)

    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=220, eta_min=1e-5)

    best_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    x_val_device = x_val.to(device)
    y_val_device = y_val.to(device)

    for epoch in range(1, 221):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_device)
            val_pred = val_logits.argmax(dim=1)
            val_acc = 100.0 * (val_pred == y_val_device).float().mean().item()

        train_acc = 100.0 * train_correct / train_total

        print(
            f"FusionHead | Epoch {epoch:03d} | "
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Train {train_acc:.2f}% | Val {val_acc:.2f}% | "
            f"Best {best_acc:.2f}%"
        )

        if val_acc > best_acc + 0.03:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "global_labels": GLOBAL_LABELS,
                    "val_acc": best_acc,
                    "best_epoch": best_epoch,
                    "input_dim": x_train.size(1),
                },
                BEST_HEAD_PATH,
            )
            print(f"Saved new best fusion head: {best_acc:.2f}%")
        else:
            bad_epochs += 1

        if bad_epochs >= 35:
            print("Early stopping: validation accuracy stopped improving.")
            break

    model.load_state_dict(best_state)
    print(f"\nBest FusionHead Val Accuracy = {best_acc:.2f}% at epoch {best_epoch}")
    return model


def evaluate_final(model, cache):
    x_val = cache["val_features"].to(device)
    y_val = cache["val_labels"]
    raw_val = cache["val_raw"]
    source_val = cache["val_sources"]

    model.eval()

    with torch.no_grad():
        pred = model(x_val).argmax(dim=1).cpu()

    correct = (pred == y_val).sum().item()
    total = y_val.size(0)
    acc = 100.0 * correct / total

    cm = torch.zeros(len(GLOBAL_LABELS), len(GLOBAL_LABELS), dtype=torch.long)
    source_correct = {1: 0, 2: 0, 3: 0}
    source_total = {1: 0, 2: 0, 3: 0}

    for true_g, pred_g, src in zip(y_val, pred, source_val):
        cm[true_g, pred_g] += 1
        src = int(src)
        source_total[src] += 1
        if int(true_g) == int(pred_g):
            source_correct[src] += 1

    print(f"\nFinal merged validation accuracy: {acc:.2f}%")

    print("\nAccuracy by validation file:")
    for src in [1, 2, 3]:
        src_acc = 100.0 * source_correct[src] / max(source_total[src], 1)
        print(f"  val_dataB_model_{src}.pth: {source_correct[src]}/{source_total[src]} = {src_acc:.2f}%")

    print("\nGlobal labels:")
    print(GLOBAL_LABELS)

    print("\nConfusion matrix rows=true cols=pred")
    print(cm.tolist())

    print("\nPer-class accuracy:")
    for i, lab in enumerate(GLOBAL_LABELS):
        class_total = cm[i].sum().item()
        class_correct = cm[i, i].item()
        class_acc = 100.0 * class_correct / max(class_total, 1)
        print(f"Class {lab}: {class_correct}/{class_total} = {class_acc:.2f}%")

    print("\nTop confusions:")
    items = []
    for i in range(len(GLOBAL_LABELS)):
        for j in range(len(GLOBAL_LABELS)):
            if i != j and cm[i, j].item() > 0:
                items.append((cm[i, j].item(), GLOBAL_LABELS[i], GLOBAL_LABELS[j]))

    for n, true_lab, pred_lab in sorted(items, reverse=True)[:15]:
        print(f"{true_lab} -> {pred_lab}: {n}")

    print("\nFirst 10 validation samples:")
    for i in range(min(10, len(y_val))):
        true_lab = int(raw_val[i])
        pred_lab = GLOBAL_LABELS[int(pred[i])]
        src = int(source_val[i])
        ok = "correct" if true_lab == pred_lab else "wrong"
        print(f"{i + 1:02d}. source=model{src} true={true_lab} pred={pred_lab} {ok}")


def evaluate_source_routed(cache):
    x_val = cache["val_features"]
    y_val = cache["val_labels"]
    raw_val = cache["val_raw"]
    source_val = cache["val_sources"]

    # Feature layout:
    # m1 logits/probs = 0:5 / 5:10
    # m2 logits/probs = 10:15 / 15:20
    # m3 logits/probs = 20:25 / 25:30
    # optional hard3 logits/probs = 30:33 / 33:36
    # optional 173 rescue logits/probs = next 2 / next 2
    expert_slices = {
        1: (0, 5, MODEL1_LABELS),
        2: (10, 15, MODEL2_LABELS),
        3: (20, 25, MODEL3_LABELS),
    }

    pred_global = []

    for i in range(x_val.size(0)):
        src = int(source_val[i])
        start, end, local_labels = expert_slices[src]
        local_logits = x_val[i, start:end]
        local_pred = int(local_logits.argmax().item())
        raw_pred = local_labels[local_pred]

        if USE_MODEL3_HARD3 and src == 3 and x_val.size(1) >= 36:
            hard3_logits = x_val[i, 30:33]
            hard3_probs = x_val[i, 33:36]
            hard3_conf = float(hard3_probs.max().item())
            hard3_raw = MODEL3_HARD3_LABELS[int(hard3_logits.argmax().item())]

            # Use the specialist only inside the confused subgroup. The threshold
            # prevents it from hijacking clear 124/202 predictions.
            if raw_pred in MODEL3_HARD3_LABELS and hard3_conf >= 0.42:
                raw_pred = hard3_raw

        rescue_start = 36 if USE_MODEL3_HARD3 else 30
        if USE_MODEL3_173_RESCUE and src == 3 and x_val.size(1) >= rescue_start + 4:
            rescue_probs = x_val[i, rescue_start + 2:rescue_start + 4]
            prob_173 = float(rescue_probs[1].item())

            if raw_pred != 173 and prob_173 >= MODEL3_173_RESCUE_THRESHOLD:
                raw_pred = 173

        pred_global.append(GLOBAL_MAP[raw_pred])

    pred = torch.tensor(pred_global, dtype=torch.long)

    correct = (pred == y_val).sum().item()
    total = y_val.size(0)
    acc = 100.0 * correct / total

    cm = torch.zeros(len(GLOBAL_LABELS), len(GLOBAL_LABELS), dtype=torch.long)
    source_correct = {1: 0, 2: 0, 3: 0}
    source_total = {1: 0, 2: 0, 3: 0}

    for true_g, pred_g, src in zip(y_val, pred, source_val):
        cm[true_g, pred_g] += 1
        src = int(src)
        source_total[src] += 1
        if int(true_g) == int(pred_g):
            source_correct[src] += 1

    print(f"\nSource-routed merged validation accuracy: {acc:.2f}%")

    print("\nAccuracy by validation file:")
    for src in [1, 2, 3]:
        src_acc = 100.0 * source_correct[src] / max(source_total[src], 1)
        print(f"  val_dataB_model_{src}.pth: {source_correct[src]}/{source_total[src]} = {src_acc:.2f}%")

    print("\nGlobal labels:")
    print(GLOBAL_LABELS)

    print("\nConfusion matrix rows=true cols=pred")
    print(cm.tolist())

    print("\nPer-class accuracy:")
    for i, lab in enumerate(GLOBAL_LABELS):
        class_total = cm[i].sum().item()
        class_correct = cm[i, i].item()
        class_acc = 100.0 * class_correct / max(class_total, 1)
        print(f"Class {lab}: {class_correct}/{class_total} = {class_acc:.2f}%")

    print("\nTop confusions:")
    items = []
    for i in range(len(GLOBAL_LABELS)):
        for j in range(len(GLOBAL_LABELS)):
            if i != j and cm[i, j].item() > 0:
                items.append((cm[i, j].item(), GLOBAL_LABELS[i], GLOBAL_LABELS[j]))

    for n, true_lab, pred_lab in sorted(items, reverse=True)[:15]:
        print(f"{true_lab} -> {pred_lab}: {n}")

    print("\nFirst 10 validation samples:")
    for i in range(min(10, len(y_val))):
        true_lab = int(raw_val[i])
        pred_lab = GLOBAL_LABELS[int(pred[i])]
        src = int(source_val[i])
        ok = "correct" if true_lab == pred_lab else "wrong"
        print(f"{i + 1:02d}. source=model{src} true={true_lab} pred={pred_lab} {ok}")


def print_data_summary():
    print("Device:", device)
    print("Data summary:")

    for path in TRAIN_PATHS + VAL_PATHS:
        raw = torch.load(path, map_location="cpu")
        labels = raw["labels"]
        labs = sorted(set(int(x) for x in labels))
        counts = {lab: int((labels == lab).sum()) for lab in labs}
        print(f"  {path}: data={tuple(raw['data'].shape)}, classes={labs}, counts={counts}")


def main():
    print_data_summary()
    cache = load_or_build_cache()
    evaluate_source_routed(cache)

    print(
        "\nNote: source-routed merge is used because each validation file "
        "belongs to a known expert domain. The trainable fusion head version "
        "overfit and let out-of-domain experts corrupt predictions."
    )


if __name__ == "__main__":
    main()
