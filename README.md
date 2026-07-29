# Multi-Model Fusion for Robust Image Classification

This repository contains a multi-domain image classification project that explores how several compact specialist image classifiers can be combined into one unified prediction system. Instead of retraining a single large model whenever new classes are added, the project trains smaller expert CNN models on local label spaces and then merges their predictions through label-space-aware fusion strategies.

The main challenge is that expert models are often confident even when an input image does not belong to their own label space. Because of this, simple averaging or direct concatenation of model outputs is unreliable. This project focuses on expert routing, local-to-global label mapping, feature-level fusion, ensemble weighting, checkpoint selection, and error-driven analysis.

## Project Highlights

- Built compact CNN expert classifiers under constrained model size and layer requirements.
- Designed a disjoint-label fusion pipeline using image-gated global, domain, and slot prediction heads.
- Designed an overlapping-label fusion pipeline using source-routed expert selection and source-level ensemble weighting.
- Improved Part A disjoint-label fusion from an early direct-fusion baseline of around 80% to 88.73% global test accuracy.
- Improved overlapping-label validation accuracy from 84.40% to 86.67% using checkpoint re-selection and targeted rescue logic.
- Implemented reproducible evaluation scripts that report accuracy, source-level performance, per-class accuracy, confusion patterns, and sample predictions.

## Repository Structure

```text
Multi-Model Fusion for Robust Image Classification/
|-- Part A/
|   |-- Model1.py
|   |-- Model2.py
|   |-- Model3.py
|   |-- Step1 merged_seed1.py
|   |-- Step2 merged seed2.py
|   |-- Step3 ensemble.py
|   |-- checkpoints_taskA/
|   `-- Task1_data/
|-- Part B/
|   |-- Final Merge.py
|   |-- Model1.py
|   |-- Model2.py
|   |-- Model3.py
|   |-- Model1 Ensemble.py
|   |-- Model3_173_Rescue.py
|   |-- TaskBTrainCommon.py
|   |-- checkpoints_taskB/
|   `-- Task2_data/
`-- legacy root scripts and validation checkpoints
```

`Part A` contains the disjoint-label fusion experiments. `Part B` contains the overlapping-label fusion experiments. Some legacy scripts remain in the project root for reference, but the organized versions are under `Part A` and `Part B`.

## Methodology

### Compact Expert Models

Each expert model is a compact convolutional neural network trained on a local five-class image classification problem. The experts output local logits and intermediate visual features. These outputs are then reused by the merged system depending on the fusion setting.

The expert models are intentionally compact rather than large backbone models. This makes the project a model-composition problem: the goal is not simply to train a bigger classifier, but to combine limited-capacity classifiers reliably.

### Part A: Disjoint-Label Fusion

Part A studies the case where expert models are trained on disjoint label spaces. A direct fusion baseline was not reliable because out-of-domain experts sometimes produced confident but incorrect predictions.

To address this, the project uses an image-gated fusion model with multiple prediction heads:

- a global classification head for the final label prediction;
- a domain head for identifying the relevant expert group;
- a slot head for learning structure inside the expert label space.

A two-seed weighted ensemble was then used to improve robustness. The final weighted ensemble achieved 88.73% global test accuracy.

### Part B: Overlapping-Label Fusion

Part B studies a harder setting where expert label spaces partially overlap. In this setting, local logits are not directly comparable because the same global class may appear in different expert contexts.

The final system uses a deterministic source-routed fusion strategy. It first selects the relevant expert source group, then combines selected checkpoint variants inside that group, maps local predictions into the global label space, and applies targeted correction for repeated error clusters.

Checkpoint re-selection and source-level weighting improved validation accuracy from 84.40% to 86.67%. The final result was 650 correct predictions out of 750 validation samples.

## Results Summary

| Fusion Setting | Main Strategy | Result |
| --- | --- | --- |
| Disjoint expert labels | Image-gated fusion with global, domain, and slot heads | 88.73% global test accuracy |
| Overlapping expert labels | Source-routed fusion with checkpoint re-selection and source-level weighting | 86.67% validation accuracy |
| Early direct fusion baseline | Direct expert-output combination | around 80% accuracy |

## Technical Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- Google Colab

## Running the Project

The code is designed to run in Google Colab or a local Python environment with PyTorch installed.

Install common dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Run Part A scripts from the Part A folder:

```bash
cd "Multi-Model Fusion for Robust Image Classification/Part A"
python "Step1 merged_seed1.py"
python "Step2 merged seed2.py"
python "Step3 ensemble.py"
```

Run the final Part B fusion script from the Part B folder:

```bash
cd "Multi-Model Fusion for Robust Image Classification/Part B"
python "Final Merge.py"
```

The evaluation scripts load saved checkpoints, run inference, print accuracy, report error patterns, and display sample predictions.

## Large File Notes

GitHub has a 100 MB file size limit. Some large training data archives and training checkpoint files are intentionally not stored in this repository. Validation data and smaller checkpoints are included where possible. If a script expects a missing training archive or large checkpoint, place the corresponding file back into the documented task folder before running full training.

Examples of omitted large files include:

- `Task2_data.zip`
- `train_dataB_model_1.pth`
- `train_dataB_model_2.pth`
- `train_dataB_model_3.pth`

## Key Takeaway

The project shows that model fusion should not be treated as a generic averaging problem. Reliable fusion depends on the relationship between expert label spaces. Disjoint-label fusion benefits from image-gated routing and auxiliary structure prediction, while overlapping-label fusion benefits from source-aware routing, checkpoint selection, and targeted error correction.

## Author

Shaoyi Lu
Master of Engineering in Computing and Software, McMaster University
