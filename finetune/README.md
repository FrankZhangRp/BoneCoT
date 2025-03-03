# Project Overview

BoneFM is a fine-tuning framework based on the DinoV2 pre-trained vit-giant2 model

# Directory Structure

- **configs/**
  - `__init__.py`
  - Various example YAML configuration files
  - `default_configs.yaml`: Default configuration file
- **data/**
  - `__init__.py`
  - `base_dataset.py`: Basic dataset implementation
  - `kfold_dataset.py`: K-fold dataset (not implemented)
  - `series_dataset.py`: Outputs file paths, barcodes, and sequence numbers (primary dataset)
  - `series3d_dataset.py`: 3D dataset for CT-CLIP
- **layers/**: DINOV2 network definitions (unmodified)
- **models/**
  - `__init__.py`: Exposes model interfaces
  - `ctvit_base.py`: CT-CLIP model
  - `knn_classification.py`: KNN model
  - `lieanr_finetune_classification.py`: Main fine-tuning model, including full parameter and partial parameter fine-tuning
  - `linear_only_classification.py`: Linear layer fine-tuning model with different learning rates and classifier layer sizes
- **run/**
  - `run_train.py`: Training entry point
- **trainer/**
  - `__init__.py`: Exposes training interfaces
  - `base_trainer.py`: Default trainer
  - `ctclip_trainer.py`: CT-CLIP trainer
  - `finetune_linear_trainer.py`: Linear layer fine-tuning trainer (no backbone fine-tuning)
  - `knn_trainer.py`: KNN trainer
  - `only_linear_trainer.py`: Linear-only fine-tuning trainer (no backbone fine-tuning, trains multiple classifiers)
- **utils/**
  - `focal_loss.py`: Focal loss implementation
  - `metric.py`: Various evaluation metrics
  - `logger.py`: Logging utilities

# Usage

## Training

The `log_display` parameter controls whether logs are displayed in the terminal. The `config-file` parameter specifies the configuration file, and the `output-dir` parameter specifies the output directory. Note that configuration file names and output directory names should be detailed and distinctive to avoid log overwriting!
```python
CUDA_VISIBLE_DEVICES=0 python run/run_trainer.py --config-file /path/yaml --output-dir ./logs/log_name --log_display
```

## BoneFM evaluation

For evaluation, simply call the EvalLinearTrainer class in `config-file`. This trainer handles the validation process for BoneFM models.

## BoneCoT evaluation

For BoneCoT evaluation, use the EvalO1LinearTrainer class in finetune/trainer/eval_trainer.py. This trainer loads all fine-tuned model weights for different tasks from the paths specified in the configuration file before performing inference. Make sure to properly configure the checkpoint paths in your configuration file before running the evaluation.

