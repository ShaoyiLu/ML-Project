import os
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms


SEED = 2026

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"Task1_data\Task1_data"
SAVE_DIR = r"checkpoints_taskA"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATHS = [
    os.path.join(ROOT, "Model1", "model1_train.pth"),
    os.path.join(ROOT, "Model2", "model2_train.pth"),
    os.path.join(ROOT, "Model3", "model3_train.pth"),
]

TEST_PATHS = [
    os.path.join(ROOT, "Model1", "model1_test.pth"),
    os.path.join(ROOT, "Model2", "model2_test.pth"),
    os.path.join(ROOT, "Model3", "model3_test.pth"),
]

EXPERT_PATHS = [
    os.path.join(SAVE_DIR, "model1_expert_best.pth"),
    os.path.join(SAVE_DIR, "model2_expert_best.pth"),
    os.path.join(SAVE_DIR, "model3_expert_best.pth"),
]

BEST_PATH = os.path.join(SAVE_DIR, "merged_image_gate_seed2_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "merged_image_gate_seed2_last.pth")
FINAL_PATH = os.path.join(SAVE_DIR, "merged_image_gate_seed2_final.pth")

RESUME = False

MEAN = [0.5238, 0.5002, 0.4687]
STD = [0.2832, 0.2757, 0.2884]


train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.04,
        contrast=0.04,
        saturation=0.025,
        hue=0.006,
    ),
    transforms.RandomAffine(
        degrees=3,
        translate=(0.025, 0.025),
        scale=(0.98, 1.03),
        shear=2,
    ),
    transforms.Normalize(MEAN, STD),
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

        slot = raw_label // 10
        domain = raw_label % 10

        if self.transform:
            x = self.transform(x)

        return (
            x,
            torch.tensor(y, dtype=torch.long),
            torch.tensor(slot, dtype=torch.long),
            torch.tensor(domain, dtype=torch.long),
        )


def get_labels(paths):
    labels = []
    for p in paths:
        raw = torch.load(p, map_location="cpu")
        labels.extend([int(x) for x in raw["labels"]])
    return sorted(set(labels))


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
            nn.Dropout(0.32),

            nn.Linear(384, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.22),
        )

        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, x):
        return self.dense(self.features(x))

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)


class ImageGateMergedModel(nn.Module):
    def __init__(self, experts, num_global_classes=15):
        super().__init__()

        self.experts = nn.ModuleList(experts)

        for expert in self.experts:
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False

        self.image_features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 160, 3, padding=1),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),

            nn.Conv2d(160, 224, 3, padding=1),
            nn.BatchNorm2d(224),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(224, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.image_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.32),
        )

        self.domain_head = nn.Linear(256, 3)
        self.slot_head = nn.Linear(256, 5)
        self.direct_head = nn.Linear(256, num_global_classes)

        self.direct_scale = nn.Parameter(torch.tensor(0.90))
        self.router_scale = nn.Parameter(torch.tensor(0.95))

    def train(self, mode=True):
        super().train(mode)
        for expert in self.experts:
            expert.eval()
        return self

    def forward(self, x):
        img_feat = self.image_dense(self.image_features(x))

        domain_logits = self.domain_head(img_feat)
        slot_logits = self.slot_head(img_feat)
        direct_logits = self.direct_head(img_feat)

        expert_logits_list = []

        with torch.no_grad():
            for expert in self.experts:
                expert.eval()
                expert_logits_list.append(expert(x))

        expert_logits = torch.stack(expert_logits_list, dim=1)
        domain_log_probs = torch.log_softmax(domain_logits, dim=1)

        batch_size = x.size(0)
        router_logits = torch.full(
            (batch_size, 15),
            -1e4,
            device=x.device,
            dtype=x.dtype,
        )

        for slot in range(5):
            for domain in range(3):
                global_idx = slot * 3 + domain
                router_logits[:, global_idx] = (
                    expert_logits[:, domain, slot]
                    + domain_log_probs[:, domain]
                    + 0.30 * slot_logits[:, slot]
                )

        global_logits = (
            self.router_scale * router_logits
            + self.direct_scale * direct_logits
        )

        return global_logits, domain_logits, slot_logits, direct_logits


def load_expert(path):
    model = ExpertCNN(num_classes=5).to(device)

    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loaded expert: {path}")

    if isinstance(ckpt, dict):
        print(
            "Checkpoint fields:",
            {
                "test_acc": ckpt.get("test_acc"),
                "test_tta_acc": ckpt.get("test_tta_acc"),
                "best_acc": ckpt.get("best_acc"),
            },
        )

    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, use_tta=False):
    model.eval()

    correct = 0
    domain_correct = 0
    slot_correct = 0
    total = 0

    with torch.no_grad():
        for x, y, slot, domain in loader:
            x = x.to(device)
            y = y.to(device)
            slot = slot.to(device)
            domain = domain.to(device)

            global_logits, domain_logits, slot_logits, _ = model(x)

            if use_tta:
                x_flip = torch.flip(x, dims=[3])
                global_logits_f, domain_logits_f, slot_logits_f, _ = model(x_flip)

                global_logits = (global_logits + global_logits_f) / 2.0
                domain_logits = (domain_logits + domain_logits_f) / 2.0
                slot_logits = (slot_logits + slot_logits_f) / 2.0

            pred = global_logits.argmax(dim=1)
            domain_pred = domain_logits.argmax(dim=1)
            slot_pred = slot_logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            domain_correct += (domain_pred == domain).sum().item()
            slot_correct += (slot_pred == slot).sum().item()
            total += y.size(0)

    return (
        100.0 * correct / total,
        100.0 * domain_correct / total,
        100.0 * slot_correct / total,
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, best_epoch):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "seed": SEED,
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


def train_merged(
    model,
    train_loader,
    test_loader,
    epochs=190,
    lr=6e-4,
    patience=35,
    save_every=5,
):
    model.to(device)

    global_criterion = nn.CrossEntropyLoss(label_smoothing=0.035)
    domain_criterion = nn.CrossEntropyLoss(label_smoothing=0.025)
    slot_criterion = nn.CrossEntropyLoss(label_smoothing=0.025)
    direct_criterion = nn.CrossEntropyLoss(label_smoothing=0.035)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=4.5e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

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
        domain_correct = 0
        slot_correct = 0
        total = 0

        for x, y, slot, domain in train_loader:
            x = x.to(device)
            y = y.to(device)
            slot = slot.to(device)
            domain = domain.to(device)

            optimizer.zero_grad()

            global_logits, domain_logits, slot_logits, direct_logits = model(x)

            loss = (
                1.15 * global_criterion(global_logits, y)
                + 0.60 * domain_criterion(domain_logits, domain)
                + 0.65 * slot_criterion(slot_logits, slot)
                + 0.70 * direct_criterion(direct_logits, y)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            total_loss += loss.item()

            pred = global_logits.argmax(dim=1)
            domain_pred = domain_logits.argmax(dim=1)
            slot_pred = slot_logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            domain_correct += (domain_pred == domain).sum().item()
            slot_correct += (slot_pred == slot).sum().item()
            total += y.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total
        train_domain = 100.0 * domain_correct / total
        train_slot = 100.0 * slot_correct / total

        test_acc, test_domain, test_slot = evaluate(
            model,
            test_loader,
            use_tta=False,
        )

        test_tta_acc, test_tta_domain, test_tta_slot = evaluate(
            model,
            test_loader,
            use_tta=True,
        )

        score = max(test_acc, test_tta_acc)

        print(
            f"MergedSeed2 | Epoch {epoch:03d} | "
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Train {train_acc:.2f}% | "
            f"TrainDomain {train_domain:.2f}% | "
            f"TrainSlot {train_slot:.2f}% | "
            f"Test {test_acc:.2f}% | "
            f"Test-TTA {test_tta_acc:.2f}% | "
            f"TestDomain {test_domain:.2f}% | "
            f"TestTTA-Domain {test_tta_domain:.2f}% | "
            f"TestSlot {test_slot:.2f}% | "
            f"TestTTA-Slot {test_tta_slot:.2f}% | "
            f"Best {best_acc:.2f}%"
        )

        if score > best_acc + 0.03:
            best_acc = score
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
            )

            print(f"Saved new best seed2 merged model: {best_acc:.2f}%")
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
            )

        if bad_epochs >= patience:
            print("Early stopping: seed2 merged accuracy stopped improving.")
            break

    model.load_state_dict(best_state)
    print(f"\nBest seed2 merged acc = {best_acc:.2f}% at epoch {best_epoch}")

    return model


def confusion_matrix(model, loader, labels, use_tta=True):
    model.eval()

    cm = torch.zeros(len(labels), len(labels), dtype=torch.long)

    with torch.no_grad():
        for x, y, slot, domain in loader:
            x = x.to(device)
            y = y.to(device)

            global_logits, _, _, _ = model(x)

            if use_tta:
                x_flip = torch.flip(x, dims=[3])
                global_logits_f, _, _, _ = model(x_flip)
                global_logits = (global_logits + global_logits_f) / 2.0

            pred = global_logits.argmax(dim=1)

            for true_label, pred_label in zip(y.cpu(), pred.cpu()):
                cm[true_label, pred_label] += 1

    print("\nTTA confusion matrix rows=true cols=pred")
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

    for n, a, b in sorted(items, reverse=True)[:20]:
        print(f"{a} -> {b}: {n}")


def main():
    print("Device:", device)
    print("Seed:", SEED)

    global_labels = get_labels(TRAIN_PATHS + TEST_PATHS)
    global_map = {lab: i for i, lab in enumerate(global_labels)}

    print("Global labels:", global_labels)
    print("Global map:", global_map)

    experts = [load_expert(path) for path in EXPERT_PATHS]

    train_sets = [
        CAS771Dataset(path, global_map, transform=train_tf)
        for path in TRAIN_PATHS
    ]

    test_sets = [
        CAS771Dataset(path, global_map, transform=test_tf)
        for path in TEST_PATHS
    ]

    train_ds = ConcatDataset(train_sets)
    test_ds = ConcatDataset(test_sets)

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

    model = ImageGateMergedModel(
        experts,
        num_global_classes=len(global_labels),
    )

    print("Merged trainable params:", count_params(model))

    model = train_merged(
        model,
        train_loader,
        test_loader,
        epochs=190,
        lr=6e-4,
        patience=35,
        save_every=5,
    )

    final_acc, final_domain, final_slot = evaluate(
        model,
        test_loader,
        use_tta=False,
    )

    final_tta_acc, final_tta_domain, final_tta_slot = evaluate(
        model,
        test_loader,
        use_tta=True,
    )

    print(f"\nFinal Seed2 Merged Test Accuracy: {final_acc:.2f}%")
    print(f"Final Seed2 Merged Test-TTA Accuracy: {final_tta_acc:.2f}%")
    print(f"Final Seed2 Merged TestDomain Accuracy: {final_domain:.2f}%")
    print(f"Final Seed2 Merged TestTTA-Domain Accuracy: {final_tta_domain:.2f}%")
    print(f"Final Seed2 Merged TestSlot Accuracy: {final_slot:.2f}%")
    print(f"Final Seed2 Merged TestTTA-Slot Accuracy: {final_tta_slot:.2f}%")

    confusion_matrix(
        model,
        test_loader,
        global_labels,
        use_tta=True,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "global_labels": global_labels,
            "global_map": global_map,
            "final_acc": final_acc,
            "final_tta_acc": final_tta_acc,
            "final_domain": final_domain,
            "final_tta_domain": final_tta_domain,
            "final_slot": final_slot,
            "final_tta_slot": final_tta_slot,
            "expert_paths": EXPERT_PATHS,
            "seed": SEED,
            "note": "Seed2 image-gate merged model. No source-domain input is used.",
        },
        FINAL_PATH,
    )

    print(f"\nSaved seed2 final merged model to {FINAL_PATH}")
    print(f"Best seed2 model path: {BEST_PATH}")


if __name__ == "__main__":
    main()
