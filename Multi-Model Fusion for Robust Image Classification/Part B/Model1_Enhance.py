import os

import torch
from torch.utils.data import DataLoader

from TaskBTrainCommon import (
    ExpertCNN,
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


set_seed(771)

ROOT = r"Task2_data"
SAVE_DIR = r"checkpoints_taskB"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(ROOT, "train_dataB_model_1.pth")
VAL_PATH = os.path.join(ROOT, "val_dataB_model_1.pth")

BEST_PATH = os.path.join(SAVE_DIR, "modelB_1_expert_best_137201.pth")
LAST_PATH = os.path.join(SAVE_DIR, "modelB_1_expert_last_137201.pth")


def main():
    labels = get_labels(TRAIN_PATH)
    label_map = {lab: i for i, lab in enumerate(labels)}

    print("Device:", device)
    print("ModelB-1 enhanced labels:", labels)
    print("ModelB-1 enhanced label map:", label_map)

    train_ds = TaskBDataset(TRAIN_PATH, label_map, transform=normalize_tf)
    val_ds = TaskBDataset(VAL_PATH, label_map, transform=normalize_tf)

    # local labels:
    # 0 = 34, 1 = 137, 2 = 159, 3 = 173, 4 = 201
    # Current ensemble is weakest around 137/201, so this run becomes a complement.
    sampler = make_weighted_sampler(
        train_ds,
        boost={
            1: 1.25,
            4: 1.35,
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

    model = ExpertCNN(num_classes=5, dropout1=0.42, dropout2=0.32)
    print("Trainable parameters:", count_parameters(model))

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        labels=labels,
        label_map=label_map,
        best_path=BEST_PATH,
        last_path=LAST_PATH,
        class_weights=[0.95, 1.22, 1.00, 0.90, 1.30],
        run_name="ModelB-1-137201",
        epochs=150,
        lr=1.2e-4,
        weight_decay=9e-4,
        label_smoothing=0.035,
        patience=40,
    )

    final_acc = evaluate(model, val_loader)
    print(f"\nFinal ModelB-1-137201 Validation Accuracy: {final_acc:.2f}%")
    confusion_matrix(model, val_loader, labels)

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "label_map": label_map,
            "val_acc": final_acc,
        },
        BEST_PATH,
    )

    print(f"\nSaved ModelB-1 enhanced best model to {BEST_PATH}")


if __name__ == "__main__":
    main()
