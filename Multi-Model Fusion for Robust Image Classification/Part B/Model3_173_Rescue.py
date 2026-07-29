import os

import torch
from torch.utils.data import DataLoader

from TaskBTrainCommon import (
    ExpertCNN,
    TaskBDataset,
    count_parameters,
    device,
    make_weighted_sampler,
    normalize_tf,
    set_seed,
    train_model,
)


set_seed(1773)

ROOT = r"Task2_data"
SAVE_DIR = r"checkpoints_taskB"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(ROOT, "train_dataB_model_3.pth")
VAL_PATH = os.path.join(ROOT, "val_dataB_model_3.pth")

MODEL3_LABELS = [124, 125, 130, 173, 202]
BINARY_LABELS = ["not_173", "173"]

BEST_PATH = os.path.join(SAVE_DIR, "modelB_3_173_rescue_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "modelB_3_173_rescue_last.pth")


def binary_metrics(model, loader, thresholds):
    model.eval()

    all_true = []
    all_prob_173 = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob_173 = torch.softmax(logits, dim=1)[:, 1].cpu()

            all_true.append(y.cpu())
            all_prob_173.append(prob_173)

    true = torch.cat(all_true)
    prob_173 = torch.cat(all_prob_173)

    print("\n173 rescue threshold scan")
    print("threshold | acc | precision_173 | recall_173 | f1_173 | predicted_173")

    best_f1 = -1.0
    best_threshold = None

    for threshold in thresholds:
        pred = (prob_173 >= threshold).long()

        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        tn = int(((pred == 0) & (true == 0)).sum())

        acc = 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = 100.0 * tp / max(tp + fp, 1)
        recall = 100.0 * tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        print(
            f"{threshold:.2f}      | {acc:5.2f}% | "
            f"{precision:6.2f}%       | {recall:6.2f}%    | "
            f"{f1:6.2f}% | {tp + fp}"
        )

    pred = (prob_173 >= best_threshold).long()
    cm = torch.zeros(2, 2, dtype=torch.long)

    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1

    print(f"\nBest threshold by F1: {best_threshold:.2f}")
    print("Confusion matrix rows=true cols=pred")
    print("Labels:", BINARY_LABELS)
    print(cm.tolist())

    return best_threshold, best_f1


def main():
    # local binary labels:
    # 0 = not_173: [124, 125, 130, 202]
    # 1 = 173
    label_map = {
        124: 0,
        125: 0,
        130: 0,
        173: 1,
        202: 0,
    }

    print("Device:", device)
    print("ModelB-3 173 rescue raw labels:", MODEL3_LABELS)
    print("ModelB-3 173 rescue binary labels:", BINARY_LABELS)
    print("Raw label map:", label_map)

    train_ds = TaskBDataset(
        TRAIN_PATH,
        label_map,
        transform=normalize_tf,
        allowed_labels=MODEL3_LABELS,
    )

    val_ds = TaskBDataset(
        VAL_PATH,
        label_map,
        transform=normalize_tf,
        allowed_labels=MODEL3_LABELS,
    )

    sampler = make_weighted_sampler(
        train_ds,
        boost={
            1: 1.7,
        },
    )

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

    model = ExpertCNN(num_classes=2, dropout1=0.42, dropout2=0.32)
    print("Trainable parameters:", count_parameters(model))

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        labels=BINARY_LABELS,
        label_map=label_map,
        best_path=BEST_PATH,
        last_path=LAST_PATH,
        class_weights=[0.75, 1.85],
        run_name="ModelB-3-173-rescue",
        epochs=170,
        lr=1.0e-4,
        weight_decay=1.2e-3,
        label_smoothing=0.02,
        patience=42,
    )

    best_threshold, best_f1 = binary_metrics(
        model,
        val_loader,
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": BINARY_LABELS,
            "label_map": label_map,
            "positive_label": 173,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "mean": [0.4920, 0.4653, 0.3957],
            "std": [0.2401, 0.2301, 0.2362],
            "note": "Binary rescue model for ModelB-3: predicts whether an image is class 173.",
        },
        BEST_PATH,
    )

    print(f"\nSaved ModelB-3 173 rescue model to {BEST_PATH}")


if __name__ == "__main__":
    main()
