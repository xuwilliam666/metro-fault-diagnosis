# Metro Fault Diagnosis

This repository contains the final codebase for a metro equipment monitoring and fault diagnosis project with two complementary branches:

- unsupervised anomaly detection on `MetroPT-3` using an `Anomaly Transformer`
- supervised fault classification using `LSTM-FCN`, including cross-domain transfer learning experiments

The final project conclusion is:

- transfer learning is feasible on structured bearing datasets such as `CWRU` and `Paderborn`
- transfer does not fully generalize to the real metro supervised dataset
- this is treated as a meaningful domain-gap finding rather than a code failure
- the final system demonstration uses lightweight decision-level fusion to produce three system states:
  - `Normal`
  - `Warning`
  - `Fault`

## Repository Structure

```text
metro-fault-diagnosis/
├── data/
│   ├── raw/                  # raw datasets
│   └── processed/            # processed MetroPT / bogie splits
├── checkpoints/             # saved model weights
├── logs/                    # metrics, curves, figures, summaries
├── scripts/                 # preprocessing scripts
└── src/
    ├── data/                # dataset loaders
    ├── models/              # LSTM-FCN and Anomaly Transformer
    └── training/            # training, transfer, integration scripts
```

## Main Datasets

### MetroPT-3
- task: unsupervised anomaly detection
- model: `Anomaly Transformer`
- final use in project: anomaly branch of the integrated system

### Metro supervised vibration dataset
- task: supervised binary classification
- labels:
  - `0 = Normal`
  - `1 = Failure`
- final use in project: real-world target-domain transfer analysis

### CWRU
- task: supervised bearing fault classification
- final use in project: source-domain pretraining for transfer learning

### Paderborn
- task: supervised bearing fault classification
- final use in project: structured target-domain transfer evaluation

### Bogie dataset
- task: supervised multi-class classification
- final use in project: auxiliary classification experiments

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

The code automatically uses CUDA when available.

## Core Models

### LSTM-FCN
Implemented in [`src/models/lstm_fcn.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/models/lstm_fcn.py).

The model exposes:

- `forward_features(x)`
- `forward_head(feat)`
- `forward(x)`

This supports both normal supervised training and transfer-learning workflows.

### Anomaly Transformer
Implemented in [`src/models/anomaly_transformer.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/models/anomaly_transformer.py).

Used for MetroPT point-level anomaly scoring.

## Recommended Workflow

### 1. Prepare MetroPT

```bash
python3 scripts/prepare_metropt3.py
```

This creates:

- `data/processed/MetroPT/train.npy`
- `data/processed/MetroPT/test.npy`
- `data/processed/MetroPT/test_label.npy`

### 2. Train the anomaly branch

```bash
python3 -m src.training.train_metropt
```

Outputs include:

- `logs/metropT_test_window_scores.npy`
- `logs/metropT_test_point_scores.npy`
- `logs/metropT_best_metrics.txt`

### 3. Train Metro supervised baseline

```bash
python3 -m src.training.train_metrodataset \
  --data_root data/raw/MetroDataset
```

This uses the leakage-fixed data pipeline in [`src/data/metro_dataset.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/data/metro_dataset.py).

### 4. Train CWRU source model

Example for the final clean transfer pair:

```bash
python3 -m src.training.train_cwru \
  --label_mode inner_outer \
  --data_root data/raw/CWRU/12k_DE
```

This produces:

- `checkpoints/best_cwru_inner_outer_lstm_fcn.pt`

### 5. Train Paderborn baseline

Example final structured target setting:

```bash
python3 -m src.training.train_paderborn \
  --task_mode inner_outer \
  --data_root data/raw/Paderborn/archive-2 \
  --include_conditions N15_M07_F10
```

### 6. Run transfer learning on Paderborn

```bash
python3 -m src.training.finetune_paderborn_from_cwru \
  --cwru_ckpt checkpoints/best_cwru_inner_outer_lstm_fcn.pt \
  --task_mode inner_outer \
  --mode full \
  --data_root data/raw/Paderborn/archive-2 \
  --include_conditions N15_M07_F10
```

For few-shot experiments, set:

- `--train_fraction 0.25`
- `--train_fraction 0.10`

### 7. Generate the final integration demo

```bash
python3 -m src.training.run_integration_demo
```

Recommended final-report style example:

```bash
python3 -m src.training.run_integration_demo \
  --demo_rise_window 2500 \
  --confidence_threshold 0.55 \
  --min_warning_len 50 \
  --min_fault_len 300 \
  --focus_window 20000
```

Outputs:

- `logs/integration_demo/integration_result.pdf`
- `logs/integration_demo/integration_result.png`
- `logs/integration_demo/integration_summary.json`
- `logs/integration_demo/integration_state_sequence.csv`

## Transfer Learning Summary

Two transfer settings were investigated:

### CWRU -> Metro supervised dataset
- result: did not generalize well
- interpretation: realistic domain gap and real-world signal complexity

### CWRU -> Paderborn
- result: transfer was feasible and stable on structured bearing data
- however, transfer did not consistently outperform strong target-only baselines

This project therefore keeps:

- `transfer learning` as an important experimental study
- `MetroPT anomaly detection` as the main temporal monitoring branch
- `decision-level fusion` as the final integrated system

## Final Integration Logic

The final integrated system is not end-to-end jointly trained.
It uses decision-level fusion:

- if `anomaly_score < T` -> `Normal`
- if `anomaly_score >= T` and `classifier_confidence < P` -> `Warning`
- else -> `Fault`

Important note:

- the anomaly score comes from a real trained `MetroPT` anomaly model
- if no naturally aligned classifier-confidence time series is available on the same MetroPT time axis, the integration script uses a transparent deterministic proxy confidence progression for system demonstration
- this is intended for final system visualization and report explanation, not as a new benchmark claim

## Main Scripts

### Data
- [`scripts/prepare_metropt3.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/scripts/prepare_metropt3.py)
- [`scripts/split_bogie_dataset1.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/scripts/split_bogie_dataset1.py)

### Training
- [`src/training/train_metropt.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/train_metropt.py)
- [`src/training/train_metrodataset.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/train_metrodataset.py)
- [`src/training/train_cwru.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/train_cwru.py)
- [`src/training/train_paderborn.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/train_paderborn.py)
- [`src/training/finetune_metro_from_cwru.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/finetune_metro_from_cwru.py)
- [`src/training/finetune_paderborn_from_cwru.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/finetune_paderborn_from_cwru.py)

### Utilities
- [`src/training/transfer_utils.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/transfer_utils.py)
- [`src/training/extract_metro_features.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/extract_metro_features.py)
- [`src/training/run_integration_demo.py`](/Users/xudingwei/Desktop/metro-fault-diagnosis/src/training/run_integration_demo.py)

## Final Report Artifacts

Recommended files to use in the final report:

- anomaly detection metric summary:
  - `logs/metropT_best_metrics.txt`
- final integration figure:
  - `logs/integration_demo/integration_result.pdf`
- integration system summary:
  - `logs/integration_demo/integration_summary.json`
- experiment setup notes:
  - `logs/section4_experimental_setup_notes.txt`

## Notes

- The Metro supervised pipeline was corrected to avoid leakage:
  - split raw sequences by time first
  - compute normalization statistics on train only
  - generate windows separately inside train/val/test
- CWRU and Paderborn loaders both support file-level splits and optional balanced sampling.
- Some raw `.mat` files may be unreadable; the loaders skip them with warnings.

