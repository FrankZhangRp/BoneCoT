# Anonymous Repository for BoneFM and BoneCoT

This repository contains the implementation of BoneFM and BoneCoT for bone image analysis. The code is organized into two main folders:

## Repository Structure

- `pretrain/`: Official BoneFM pretraining code based on DINOv2
- `finetune/`: Code for fine-tuning and testing BoneFM and BoneCoT models

## Pretrain

The `pretrain` folder contains the official BoneFM pretraining code based on DINOv2. We made simple modifications to the dataset part and implemented the code using PyTorch version > 2.1 with dinov2-patch.

### Requirements

- PyTorch > 2.1
- DINOv2 dependencies

## Finetune

The `finetune` folder contains code for fine-tuning and testing both BoneFM and BoneCoT models.

### Data Format

- **BoneFM**: Accepts labels in either `.txt` or `.csv` format
- **BoneCoT**: Requires labels in `.csv` format with the following structure:
  - First column: `image_path` - Path to the image file
  - Second column: `label` - Main task label
  - Subsequent columns: Additional task-related labels/features

### Configuration

Example configuration files are provided in the `finetune/configs/` directory:
- `finetune.yaml`: Basic fine-tuning configuration
- `bonecot_eval.yaml`: Configuration for BoneCoT evaluation

### Usage

1. Prepare your dataset according to the required format
2. Configure the appropriate YAML file with your dataset paths and training parameters
3. Run the training/evaluation scripts
