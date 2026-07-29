import os
import copy
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

TRAIN_PATH = os.path.join(ROOT, "train_dataB_model_3.pth")
VAL_PATH = os.path.join(ROOT, "val_dataB_model_3.pth")

BEST_PATH = os.path.join(SAVE_DIR, "modelB_3_expert_best_rebuild.pth")
LAST_PATH = os.path.join(SAVE_DIR, "modelB_3_expert_last_rebuild.pth")

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

    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_hard_sampler(dataset):
    mapped_labels = [
        dataset.label_map[int(raw_label)]
        for raw_label in dataset.labels
    ]

    counts = Counter(mapped_labels)

    # local labels:
    # 0 = 124, 1 = 125, 2 = 130, 3 = 173, 4 = 202
    boost = {
        0: 0.90,
        1: 1.05,
        2: 1.45,
        3: 1.80,
        4: 0.90,
    }

    weights = []
    for y in mapped_labels:
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

            logits = model(x)
            pred = logits.argmax(dim=1)

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
    epochs=180,
    lr=2e-4,
    patience=45,
    save_every=5,
):
    model.to(device)

    # local labels:
    # 0 = 124, 1 = 125, 2 = 130, 3 = 173, 4 = 202
    class_weights = torch.tensor(
        [0.95, 1.05, 1.25, 1.45, 0.95],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.03,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            total_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total
        val_acc = evaluate(model, val_loader)

        print(
            f"ModelB-3-rebuild | Epoch {epoch:03d} | "
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
                BEST_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                best_epoch,
                labels,
                label_map,
            )

            print(f"Saved new best ModelB-3-rebuild: {best_acc:.2f}%")
        else:
            bad_epochs += 1

        if epoch % save_every == 0:
            save_checkpoint(
                LAST_PATH,
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
    print(f"\nBest ModelB-3-rebuild Val Accuracy = {best_acc:.2f}% at epoch {best_epoch}")

    return model


def confusion_matrix(model, loader, labels):
    model.eval()

    cm = torch.zeros(len(labels), len(labels), dtype=torch.long)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)

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


def main():
    labels = get_labels(TRAIN_PATH)
    label_map = {lab: i for i, lab in enumerate(labels)}

    print("Device:", device)
    print("ModelB-3 labels:", labels)
    print("ModelB-3 label map:", label_map)

    train_ds = TaskBDataset(
        TRAIN_PATH,
        label_map,
        transform=normalize_tf,
    )

    val_ds = TaskBDataset(
        VAL_PATH,
        label_map,
        transform=normalize_tf,
    )

    sampler = make_hard_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = CompactExpertCNN(num_classes=5)

    print("Trainable parameters:", count_parameters(model))

    model = train_model(
        model,
        train_loader,
        val_loader,
        labels,
        label_map,
        epochs=180,
        lr=2e-4,
        patience=45,
        save_every=5,
    )

    final_acc = evaluate(model, val_loader)

    print(f"\nFinal ModelB-3-rebuild Validation Accuracy: {final_acc:.2f}%")

    confusion_matrix(model, val_loader, labels)

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "label_map": label_map,
            "val_acc": final_acc,
            "mean": MEAN,
            "std": STD,
        },
        BEST_PATH,
    )

    print(f"\nSaved ModelB-3-rebuild best model to {BEST_PATH}")


if __name__ == "__main__":
    main()
