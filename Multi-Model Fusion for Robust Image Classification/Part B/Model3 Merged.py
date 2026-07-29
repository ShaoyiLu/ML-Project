import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"Task2_data"
SAVE_DIR = r"checkpoints_taskB"

VAL_PATH = os.path.join(ROOT, "val_dataB_model_3.pth")

LOWLR_PATH = os.path.join(SAVE_DIR, "modelB_3_expert_best_lowlr.pth")
TRANSFER_PATH = os.path.join(SAVE_DIR, "modelB_3_expert_best_transfer.pth")
REBUILD_PATH = os.path.join(SAVE_DIR, "modelB_3_expert_best_rebuild.pth")

MEAN = [0.4920, 0.4653, 0.3957]
STD = [0.2401, 0.2301, 0.2362]

normalize_tf = transforms.Normalize(MEAN, STD)


class TaskBDataset(Dataset):
    def __init__(self, path, label_map, transform=None):
        raw = torch.load(path, map_location="cpu")
        self.data = raw["data"]
        self.labels = raw["labels"]
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].float() / 255.0

        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)

        if self.transform is not None:
            x = self.transform(x)

        raw_label = int(self.labels[idx])
        y = self.label_map[raw_label]

        return x, torch.tensor(y, dtype=torch.long)


def get_labels(path):
    raw = torch.load(path, map_location="cpu")
    return sorted(set(int(x) for x in raw["labels"]))


class ExpertCNN(nn.Module):
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
        feat = self.dense(self.features(x))
        return self.classifier(feat)


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


def load_model(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model


def evaluate_ensemble(models, weights, loader, labels):
    correct = 0
    total = 0

    cm = torch.zeros(len(labels), len(labels), dtype=torch.long)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits_sum = None

            for model, weight in zip(models, weights):
                logits = model(x) * weight

                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

            pred = logits_sum.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            for true_label, pred_label in zip(y.cpu(), pred.cpu()):
                cm[true_label, pred_label] += 1

    acc = 100.0 * correct / total
    return acc, cm


def print_confusion(cm, labels):
    print("\nConfusion matrix rows=true cols=pred")
    print("Labels:", labels)
    print(cm.tolist())

    for i, lab in enumerate(labels):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        acc = 100.0 * correct / max(total, 1)
        print(f"Class {lab}: {correct}/{total} = {acc:.2f}%")

    print("\nTop confusions:")
    items = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j].item() > 0:
                items.append((cm[i, j].item(), labels[i], labels[j]))

    for n, true_lab, pred_lab in sorted(items, reverse=True)[:10]:
        print(f"{true_lab} -> {pred_lab}: {n}")


def main():
    labels = get_labels(VAL_PATH)
    label_map = {lab: i for i, lab in enumerate(labels)}

    print("Device:", device)
    print("ModelB-3 ensemble labels:", labels)
    print("ModelB-3 ensemble label map:", label_map)

    val_ds = TaskBDataset(
        VAL_PATH,
        label_map,
        transform=normalize_tf,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    lowlr_model = load_model(
        ExpertCNN(num_classes=5),
        LOWLR_PATH,
    )

    transfer_model = load_model(
        ExpertCNN(num_classes=5),
        TRANSFER_PATH,
    )

    rebuild_model = load_model(
        CompactExpertCNN(num_classes=5),
        REBUILD_PATH,
    )

    models = [
        lowlr_model,
        transfer_model,
        rebuild_model,
    ]

    weights = [
        0.6,
        0.6,
        0.8,
    ]

    print("Loaded checkpoints:")
    print("  lowlr:", LOWLR_PATH)
    print("  transfer:", TRANSFER_PATH)
    print("  rebuild:", REBUILD_PATH)
    print("Weights: lowlr=0.6, transfer=0.6, rebuild=0.8")

    acc, cm = evaluate_ensemble(
        models,
        weights,
        val_loader,
        labels,
    )

    print(f"\nModelB-3 Ensemble Validation Accuracy: {acc:.2f}%")

    print_confusion(cm, labels)


if __name__ == "__main__":
    main()
