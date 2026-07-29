import os

import torch
from torch.utils.data import DataLoader

from TaskBTrainCommon import (
    CompactExpertCNN,
    TaskBDataset,
    confusion_matrix,
    count_parameters,
    device,
    evaluate,
    get_labels,
    make_weighted_sampler,
    normalize_tf,
    set_seed,
    train_model,
)


set_seed(773)

ROOT = r"Task2_data"
SAVE_DIR = r"checkpoints_taskB"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(ROOT, "train_dataB_model_3.pth")
VAL_PATH = os.path.join(ROOT, "val_dataB_model_3.pth")

HARD_LABELS = [125, 130, 173]

BEST_PATH = os.path.join(SAVE_DIR, "modelB_3_hard3_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, "modelB_3_hard3_last.pth")


def main():
    labels = get_labels(TRAIN_PATH, allowed_labels=HARD_LABELS)
    label_map = {lab: i for i, lab in enumerate(labels)}

    print("Device:", device)
    print("ModelB-3 hard-3 labels:", labels)
    print("ModelB-3 hard-3 label map:", label_map)

    train_ds = TaskBDataset(
        TRAIN_PATH,
        label_map,
        transform=normalize_tf,
        allowed_labels=HARD_LABELS,
    )

    val_ds = TaskBDataset(
        VAL_PATH,
        label_map,
        transform=normalize_tf,
        allowed_labels=HARD_LABELS,
    )

    # local labels:
    # 0 = 125, 1 = 130, 2 = 173
    # This model is not a full expert. It is a specialist for the confused group.
    sampler = make_weighted_sampler(
        train_ds,
        boost={
            1: 1.25,
            2: 1.55,
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

    model = CompactExpertCNN(num_classes=3)
    print("Trainable parameters:", count_parameters(model))

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        labels=labels,
        label_map=label_map,
        best_path=BEST_PATH,
        last_path=LAST_PATH,
        class_weights=[1.05, 1.25, 1.55],
        run_name="ModelB-3-hard3",
        epochs=180,
        lr=2.0e-4,
        weight_decay=1.0e-3,
        label_smoothing=0.025,
        patience=45,
    )

    final_acc = evaluate(model, val_loader)
    print(f"\nFinal ModelB-3-hard3 Validation Accuracy: {final_acc:.2f}%")
    confusion_matrix(model, val_loader, labels)

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "label_map": label_map,
            "val_acc": final_acc,
            "note": "Specialist for labels [125, 130, 173]. Use only as reranker/rerouter.",
        },
        BEST_PATH,
    )

    print(f"\nSaved ModelB-3 hard-3 specialist best model to {BEST_PATH}")


if __name__ == "__main__":
    main()
