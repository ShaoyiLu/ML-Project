import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"Task1_data\Task1_data"
SAVE_DIR = r"checkpoints_taskA"

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

MERGED_PATHS = [
    os.path.join(SAVE_DIR, "merged_image_gate_best.pth"),
    os.path.join(SAVE_DIR, "merged_image_gate_seed2_best.pth"),
]

ENSEMBLE_INFO_PATH = os.path.join(SAVE_DIR, "merged_ensemble_seed1_seed2_info.pth")

MEAN = [0.5238, 0.5002, 0.4687]
STD = [0.2832, 0.2757, 0.2884]

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
            nn.Dropout(0.35),
        )

        self.domain_head = nn.Linear(256, 3)
        self.slot_head = nn.Linear(256, 5)
        self.direct_head = nn.Linear(256, num_global_classes)

        self.direct_scale = nn.Parameter(torch.tensor(0.75))
        self.router_scale = nn.Parameter(torch.tensor(1.00))

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
                    + 0.25 * slot_logits[:, slot]
                )

        global_logits = (
            self.router_scale * router_logits
            + self.direct_scale * direct_logits
        )

        return global_logits, domain_logits, slot_logits, direct_logits


class ImageGateMergedModelSeed2(nn.Module):
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


def load_merged(path, model_class, experts, name):
    model = model_class(experts, num_global_classes=15).to(device)

    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loaded merged {name}: {path}")
    if isinstance(ckpt, dict):
        print(
            "Merged fields:",
            {
                "best_acc": ckpt.get("best_acc"),
                "final_tta_acc": ckpt.get("final_tta_acc"),
                "epoch": ckpt.get("epoch"),
                "best_epoch": ckpt.get("best_epoch"),
                "seed": ckpt.get("seed"),
            },
        )

    return model


def evaluate_single(model, loader, use_tta=False):
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


def evaluate_ensemble(models, loader, use_tta=False, weights=None):
    for model in models:
        model.eval()

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

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

            global_sum = None
            domain_sum = None
            slot_sum = None

            for weight, model in zip(weights, models):
                global_logits, domain_logits, slot_logits, _ = model(x)

                if use_tta:
                    x_flip = torch.flip(x, dims=[3])
                    global_logits_f, domain_logits_f, slot_logits_f, _ = model(x_flip)

                    global_logits = (global_logits + global_logits_f) / 2.0
                    domain_logits = (domain_logits + domain_logits_f) / 2.0
                    slot_logits = (slot_logits + slot_logits_f) / 2.0

                if global_sum is None:
                    global_sum = weight * global_logits
                    domain_sum = weight * domain_logits
                    slot_sum = weight * slot_logits
                else:
                    global_sum += weight * global_logits
                    domain_sum += weight * domain_logits
                    slot_sum += weight * slot_logits

            pred = global_sum.argmax(dim=1)
            domain_pred = domain_sum.argmax(dim=1)
            slot_pred = slot_sum.argmax(dim=1)

            correct += (pred == y).sum().item()
            domain_correct += (domain_pred == domain).sum().item()
            slot_correct += (slot_pred == slot).sum().item()
            total += y.size(0)

    return (
        100.0 * correct / total,
        100.0 * domain_correct / total,
        100.0 * slot_correct / total,
    )


def confusion_matrix_ensemble(models, loader, labels, use_tta=True, weights=None):
    for model in models:
        model.eval()

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    cm = torch.zeros(len(labels), len(labels), dtype=torch.long)

    with torch.no_grad():
        for x, y, slot, domain in loader:
            x = x.to(device)

            global_sum = None

            for weight, model in zip(weights, models):
                global_logits, _, _, _ = model(x)

                if use_tta:
                    x_flip = torch.flip(x, dims=[3])
                    global_logits_f, _, _, _ = model(x_flip)
                    global_logits = (global_logits + global_logits_f) / 2.0

                if global_sum is None:
                    global_sum = weight * global_logits
                else:
                    global_sum += weight * global_logits

            pred = global_sum.argmax(dim=1)

            for true_label, pred_label in zip(y.cpu(), pred.cpu()):
                cm[true_label, pred_label] += 1

    print("\nEnsemble TTA confusion matrix rows=true cols=pred")
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

    global_labels = get_labels(TRAIN_PATHS + TEST_PATHS)
    global_map = {lab: i for i, lab in enumerate(global_labels)}

    print("Global labels:", global_labels)
    print("Global map:", global_map)

    experts_seed1 = [load_expert(path) for path in EXPERT_PATHS]
    experts_seed2 = [load_expert(path) for path in EXPERT_PATHS]

    seed1 = load_merged(
        MERGED_PATHS[0],
        ImageGateMergedModel,
        experts_seed1,
        name="seed1",
    )

    seed2 = load_merged(
        MERGED_PATHS[1],
        ImageGateMergedModelSeed2,
        experts_seed2,
        name="seed2",
    )

    test_sets = [
        CAS771Dataset(path, global_map, transform=test_tf)
        for path in TEST_PATHS
    ]

    test_ds = ConcatDataset(test_sets)

    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nSingle model check:")

    seed1_acc, seed1_domain, seed1_slot = evaluate_single(
        seed1,
        test_loader,
        use_tta=True,
    )

    seed2_acc, seed2_domain, seed2_slot = evaluate_single(
        seed2,
        test_loader,
        use_tta=True,
    )

    print(
        f"Seed1 Test-TTA: {seed1_acc:.2f}% | "
        f"Domain {seed1_domain:.2f}% | Slot {seed1_slot:.2f}%"
    )
    print(
        f"Seed2 Test-TTA: {seed2_acc:.2f}% | "
        f"Domain {seed2_domain:.2f}% | Slot {seed2_slot:.2f}%"
    )

    print("\nEnsemble checks:")

    candidates = [
        ("0.10 seed1 + 0.90 seed2", [0.10, 0.90]),
        ("0.15 seed1 + 0.85 seed2", [0.15, 0.85]),
        ("0.30 seed1 + 0.70 seed2", [0.30, 0.70]),
        ("0.20 seed1 + 0.80 seed2", [0.20, 0.80]),
    ]

    best_name = None
    best_weights = None
    best_acc = 0.0
    best_domain = 0.0
    best_slot = 0.0

    for name, weights in candidates:
        acc, domain_acc, slot_acc = evaluate_ensemble(
            [seed1, seed2],
            test_loader,
            use_tta=True,
            weights=weights,
        )

        print(
            f"{name} -> Test-TTA {acc:.2f}% | "
            f"Domain {domain_acc:.2f}% | Slot {slot_acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            best_domain = domain_acc
            best_slot = slot_acc
            best_name = name
            best_weights = weights

    print(f"\nBest ensemble: {best_name}")
    print(f"Best Ensemble Test-TTA Accuracy: {best_acc:.2f}%")
    print(f"Best Ensemble Domain Accuracy: {best_domain:.2f}%")
    print(f"Best Ensemble Slot Accuracy: {best_slot:.2f}%")

    confusion_matrix_ensemble(
        [seed1, seed2],
        test_loader,
        global_labels,
        use_tta=True,
        weights=best_weights,
    )

    torch.save(
        {
            "ensemble_paths": MERGED_PATHS,
            "expert_paths": EXPERT_PATHS,
            "global_labels": global_labels,
            "global_map": global_map,
            "best_ensemble_name": best_name,
            "best_weights": best_weights,
            "best_acc": best_acc,
            "best_domain": best_domain,
            "best_slot": best_slot,
            "note": "Seed1 + Seed2 image-gate ensemble. No source-domain input is used.",
        },
        ENSEMBLE_INFO_PATH,
    )

    print(f"\nSaved ensemble info to {ENSEMBLE_INFO_PATH}")


if __name__ == "__main__":
    main()
