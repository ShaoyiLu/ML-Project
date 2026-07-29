import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"Task2_data"
SAVE_DIR = r"checkpoints_taskB"

VAL_PATH = os.path.join(ROOT, "val_dataB_model_1.pth")

BN_OLD_PATH = os.path.join(SAVE_DIR, "modelB_1_expert_best.pth")
BN_LOWLR_PATH = os.path.join(SAVE_DIR, "modelB_1_expert_best_lowlr.pth")
GN_PATH = os.path.join(SAVE_DIR, "modelB_1_expert_best_groupnorm.pth")

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


class ExpertCNN_BN(nn.Module):
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


class ExpertCNN_GN(nn.Module):
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
        feat = self.dense(self.features(x))
        return self.classifier(feat)


def load_model(model, path):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def evaluate_ensemble(models, loader, weights):
    correct = 0
    total = 0
    cm = torch.zeros(5, 5, dtype=torch.long)

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


def main():
    labels = get_labels(VAL_PATH)
    label_map = {lab: i for i, lab in enumerate(labels)}

    print("Device:", device)
    print("Labels:", labels)
    print("Label map:", label_map)

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

    models = []

    if os.path.exists(BN_OLD_PATH):
        models.append(("old", load_model(ExpertCNN_BN(), BN_OLD_PATH)))

    if os.path.exists(BN_LOWLR_PATH):
        models.append(("lowlr", load_model(ExpertCNN_BN(), BN_LOWLR_PATH)))

    if os.path.exists(GN_PATH):
        models.append(("groupnorm", load_model(ExpertCNN_GN(), GN_PATH)))

    print("Loaded models:", [name for name, _ in models])

    model_list = [m for _, m in models]

    weight_sets = [
        [1.0, 1.0, 1.0],
        [0.8, 1.0, 0.8],
        [1.0, 1.0, 0.6],
        [1.0, 1.2, 0.6],
        [1.2, 1.0, 0.6],
    ]

    best_acc = 0.0
    best_weights = None
    best_cm = None

    for weights in weight_sets:
        weights = weights[:len(model_list)]
        acc, cm = evaluate_ensemble(model_list, val_loader, weights)
        print(f"Weights {weights} | Ensemble Val Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_weights = weights
            best_cm = cm

    print(f"\nBest ensemble accuracy: {best_acc:.2f}%")
    print("Best weights:", best_weights)

    print_confusion(best_cm, labels)


if __name__ == "__main__":
    main()
