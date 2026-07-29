import copy
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.4920, 0.4653, 0.3957]
STD = [0.2401, 0.2301, 0.2362]
normalize_tf = transforms.Normalize(MEAN, STD)


def set_seed(seed=771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TaskBDataset(Dataset):
    def __init__(self, path, label_map, transform=None, allowed_labels=None):
        raw = torch.load(path, map_location="cpu")
        self.data = raw["data"]
        self.labels = raw["labels"]
        self.label_map = label_map
        self.transform = transform

        if allowed_labels is None:
            self.indices = list(range(len(self.labels)))
        else:
            allowed = set(int(x) for x in allowed_labels)
            self.indices = [
                i for i, y in enumerate(self.labels)
                if int(y) in allowed
            ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.data[real_idx].float() / 255.0

        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)

        if self.transform is not None:
            x = self.transform(x)

        raw_label = int(self.labels[real_idx])
        y = self.label_map[raw_label]

        return x, torch.tensor(y, dtype=torch.long)

    def mapped_labels(self):
        return [
            self.label_map[int(self.labels[i])]
            for i in self.indices
        ]


def get_labels(path, allowed_labels=None):
    raw = torch.load(path, map_location="cpu")
    labels = sorted(set(int(x) for x in raw["labels"]))

    if allowed_labels is not None:
        allowed = set(int(x) for x in allowed_labels)
        labels = [x for x in labels if x in allowed]

    return labels


class ExpertCNN(nn.Module):
    def __init__(self, num_classes=5, dropout1=0.38, dropout2=0.28):
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
            nn.Dropout(dropout1),

            nn.Linear(384, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout2),
        )

        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, x):
        return self.dense(self.features(x))

    def forward(self, x):
        return self.classifier(self.extract_features(x))


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
    def __init__(self, num_classes=3):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_weighted_sampler(dataset, boost=None):
    boost = boost or {}
    labels = dataset.mapped_labels()
    counts = Counter(labels)
    weights = []

    for y in labels:
        w = 1.0 / counts[y]
        w *= boost.get(y, 1.0)
        weights.append(w)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, best_epoch, labels, label_map):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "labels": labels,
            "label_map": label_map,
            "mean": MEAN,
            "std": STD,
        },
        path,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    labels,
    label_map,
    best_path,
    last_path,
    class_weights,
    run_name,
    epochs=140,
    lr=1.2e-4,
    weight_decay=8e-4,
    label_smoothing=0.03,
    patience=35,
    max_grad_norm=4.0,
):
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=label_smoothing,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    best_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total
        val_acc = evaluate(model, val_loader)

        print(
            f"{run_name} | Epoch {epoch:03d} | "
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Train {train_acc:.2f}% | Val {val_acc:.2f}% | "
            f"Best {best_acc:.2f}%"
        )

        if val_acc > best_acc + 0.03:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                best_epoch,
                labels,
                label_map,
            )
            print(f"Saved new best {run_name}: {best_acc:.2f}%")
        else:
            bad_epochs += 1

        if epoch % 5 == 0:
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                best_epoch,
                labels,
                label_map,
            )

        if bad_epochs >= patience:
            print("Early stopping: validation accuracy stopped improving.")
            break

    model.load_state_dict(best_state)
    print(f"\nBest {run_name} Val Accuracy = {best_acc:.2f}% at epoch {best_epoch}")
    return model


def confusion_matrix(model, loader, labels):
    model.eval()
    cm = torch.zeros(len(labels), len(labels), dtype=torch.long)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).argmax(dim=1)
            for true_label, pred_label in zip(y.cpu(), pred.cpu()):
                cm[true_label, pred_label] += 1

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
