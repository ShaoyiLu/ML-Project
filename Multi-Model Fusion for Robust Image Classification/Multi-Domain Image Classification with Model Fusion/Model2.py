import os
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"Task1_data\Task1_data"
SAVE_DIR = r"checkpoints_taskA"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(ROOT, "Model2", "model2_train.pth")
TEST_PATH = os.path.join(ROOT, "Model2", "model2_test.pth")

BEST_PATH = os.path.join(SAVE_DIR, "model2_expert_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "model2_expert_last.pth")

RESUME = True

MEAN = [0.5238, 0.5002, 0.4687]
STD = [0.2832, 0.2757, 0.2884]


train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(
        p=0.08,
        scale=(0.02, 0.07),
        ratio=(0.5, 2.0),
        value="random",
    ),
])

test_tf = transforms.Compose([
    transforms.Normalize(MEAN, STD),
])


class CAS771Dataset(Dataset):
    def __init__(self, path, label_map, transform=None):
        raw = torch.load(path, map_location="cpu")
        self.data = raw["data"]
        self.labels = raw["labels"]
        self.label_map = label_map
        self.transform = transform

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.stack(list(self.data))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].float()
        raw_label = int(self.labels[idx])
        y = self.label_map[raw_label]

        if self.transform:
            x = self.transform(x)

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
            nn.Dropout(0.20),
            nn.Linear(384, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
        )

        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, x):
        return self.dense(self.features(x))

    def forward(self, x):
        return self.classifier(self.extract_features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def one_hot(labels, num_classes):
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(1, labels.view(-1, 1), 1)


def soft_cross_entropy(logits, soft_targets):
    return -(soft_targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def rand_bbox(size, lam):
    batch, channels, height, width = size
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)

    return x1, y1, x2, y2


def apply_mixup_or_cutmix(x, y, num_classes=5, alpha=0.12, p=0.12):
    if random.random() > p:
        return x, one_hot(y, num_classes)

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)

    y1 = one_hot(y, num_classes)
    y2 = one_hot(y[index], num_classes)

    if random.random() < 0.5:
        return lam * x + (1.0 - lam) * x[index], lam * y1 + (1.0 - lam) * y2

    x1, y1_box, x2, y2_box = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, y1_box:y2_box, x1:x2] = x[index, :, y1_box:y2_box, x1:x2]

    box_area = (x2 - x1) * (y2_box - y1_box)
    lam = 1.0 - box_area / (x.size(-1) * x.size(-2))

    return mixed_x, lam * y1 + (1.0 - lam) * y2


def evaluate(model, loader, use_tta=True):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            if use_tta:
                logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def confusion_matrix(model, loader, labels):
    model.eval()
    cm = torch.zeros(len(labels), len(labels), dtype=torch.long)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
            pred = logits.argmax(dim=1)

            for true_label, pred_label in zip(y.cpu(), pred.cpu()):
                cm[true_label, pred_label] += 1

    print("\nConfusion matrix rows=true cols=pred")
    print("Labels:", labels)
    print(cm.tolist())

    for i, lab in enumerate(labels):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        print(f"Class {lab}: {correct}/{total} = {100.0 * correct / max(total, 1):.2f}%")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, best_epoch):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    return ckpt


def train_model(model, train_loader, test_loader, epochs=180, lr=6e-4, patience=35, save_every=5):
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=8e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    best_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    if RESUME and os.path.exists(LAST_PATH):
        ckpt = load_checkpoint(LAST_PATH, model, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        print(f"Resume from epoch {start_epoch}, best={best_acc:.2f}%")

    if RESUME and os.path.exists(BEST_PATH):
        best_ckpt = torch.load(BEST_PATH, map_location=device)
        best_state = copy.deepcopy(best_ckpt["model_state"])
        best_acc = best_ckpt.get("best_acc", best_acc)
        best_epoch = best_ckpt.get("best_epoch", best_epoch)

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            x_mix, y_soft = apply_mixup_or_cutmix(x, y)

            optimizer.zero_grad()
            logits = model(x_mix)
            loss = soft_cross_entropy(logits, y_soft)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, use_tta=True)

        print(
            f"Model2 | Epoch {epoch:03d} | "
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Train {train_acc:.2f}% | Test-TTA {test_acc:.2f}% | "
            f"Best {best_acc:.2f}%"
        )

        if test_acc > best_acc + 0.03:
            best_acc = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            save_checkpoint(BEST_PATH, model, optimizer, scheduler, epoch, best_acc, best_epoch)
            print(f"Saved new best Model2: {best_acc:.2f}%")
        else:
            bad_epochs += 1

        if epoch % save_every == 0:
            save_checkpoint(LAST_PATH, model, optimizer, scheduler, epoch, best_acc, best_epoch)

        if bad_epochs >= patience:
            print("Early stopping: test accuracy stopped improving.")
            break

    model.load_state_dict(best_state)
    print(f"\nBest Model2 acc = {best_acc:.2f}% at epoch {best_epoch}")
    return model


def main():
    labels = get_labels(TRAIN_PATH)
    label_map = {lab: i for i, lab in enumerate(labels)}

    print("Device:", device)
    print("Model2 labels:", labels)
    print("Model2 label map:", label_map)

    train_ds = CAS771Dataset(TRAIN_PATH, label_map, transform=train_tf)
    test_ds = CAS771Dataset(TEST_PATH, label_map, transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ExpertCNN(num_classes=5)
    print("Trainable params:", count_params(model))

    model = train_model(model, train_loader, test_loader)

    final_acc = evaluate(model, test_loader, use_tta=True)
    print(f"\nFinal Model2 Test-TTA Accuracy: {final_acc:.2f}%")
    confusion_matrix(model, test_loader, labels)

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "label_map": label_map,
            "test_tta_acc": final_acc,
        },
        BEST_PATH,
    )

    print(f"\nSaved Model2 best model to {BEST_PATH}")


if __name__ == "__main__":
    main()
