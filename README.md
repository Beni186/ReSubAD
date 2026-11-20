# ReSubAD: Residual Subspace Modeling for Zero-Shot Anomaly Detection

Official code for **ReSubAD: Residual Subspace Modeling for Zero-Shot Anomaly Detection** (WACV 2027).

> Paper: []

## Overview

ReSubAD detects anomalies using visual features from a frozen CLIP backbone. It first retrieves normal references to form residual features, then scores their projection onto a learned anomaly subspace. The repository includes code for constructing a normal reference bank, training the anomaly detection head, and evaluating image-level detection and pixel-level localization.

## Repository structure

- `train.py`: construct the normal reference bank and train the model.
- `test.py`: run inference and compute evaluation metrics.
- `run.sh`: example training and evaluation commands.
- `model/`: residual subspace model and related utilities.
- `dataset/`: dataset loaders.
- `AnomalyCLIP_lib/`: CLIP-related model implementation.

## Data preparation

Set `EXP_CA_DATA_ROOT` to the directory containing the processed datasets:

```bash
export EXP_CA_DATA_ROOT=/path/to/datasets
```

For example, the loaders expect MVTec AD and VisA under:

```text
/path/to/datasets/
├── mvtec_anomaly_detection/
│   └── meta.json
└── VisA_20220922/
    └── meta.json
```

Each dataset's `meta.json` must provide the image paths, anomaly labels, and mask paths used by the corresponding loader. See `dataset/base_dataset.py` and the dataset-specific loaders for the expected format and directory names.

## Training

The following command follows the configuration in `run.sh`:

```bash
python train.py \
  --dataset visa \
  --pretrained AnomalyCLIP_lib/ViT-L-14-336px.pt \
  --save_path ckpt/visa \
  --epochs 2 \
  --batch_size 4 \
```

The command saves the normal reference bank and epoch checkpoints under `ckpt/visa/`.

## Evaluation

For example, evaluate the VisA-trained checkpoint on MVTec AD:

```bash
python test.py \
  --dataset mvtec \
  --pretrained AnomalyCLIP_lib/ViT-L-14-336px.pt \
  --checkpoint ckpt/visa/epoch_2.pth \
  --save_path ckpt/visa/mvtec_test \
  --batch_size 4 \
```

The evaluation reports per-class and average image-level AUROC/AP and pixel-level AUROC/AP/AUPRO. Metrics can also be saved as JSON using `--save_metrics`.
