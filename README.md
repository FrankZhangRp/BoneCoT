# BoneCoT: Multi-center validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought

[![GitHub stars](https://img.shields.io/github/stars/FrankZhangRp/BoneCoT?style=flat-square)](https://github.com/FrankZhangRp/BoneCoT/stargazers)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=flat-square)](LICENSE)

Official code release for the BoneCoT paper.

This repository provides:

- `BoneFM` pretraining code built on top of DINOv2
- Fine-tuning and evaluation code for BoneFM and BoneCoT
- Public notebook templates for downstream inference

## Overview

### Included in this public release

- Source code
- Public notebook templates
- The released `BoneFM.pth` pretrained backbone checkpoint

### Not included in this public release

- Training, validation, or test datasets
- Task-specific fine-tuned checkpoints
- Review-only demo assets from the anonymous submission version

To run local evaluation, replace the placeholder paths in the notebooks or config files with your own private assets.

## Repository Structure

| Path | Description |
| --- | --- |
| `pretrain/` | BoneFM pretraining pipeline based on DINOv2 |
| `finetune/` | Fine-tuning and evaluation code for BoneFM and BoneCoT |
| `scripts/` | 5-fold notebook templates for task-specific evaluation |
| `BoneCoT_inference.ipynb` | Local BoneCoT inference template |

## Quick Start

### Environment

Recommended setup:

- OS: Ubuntu 22.04
- GPU: NVIDIA GPU with >=24 GB VRAM
- CUDA: 12.4
- Python: 3.9
- Conda: Anaconda or Miniconda

Clone the repository:

```sh
git clone https://github.com/FrankZhangRp/BoneCoT.git
cd BoneCoT
```

Create the main environment:

```sh
conda create -n bonecot python=3.9 -y
conda activate bonecot
```

Install PyTorch:

```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install project dependencies:

```sh
pip install -r requirements.txt
```

Optional: install the DINOv2 pretraining environment:

```sh
conda env create -f pretrain/conda.yaml
conda activate dinov2_new
cd pretrain
pip install -e .
```

## BoneFM Pretrained Model Download

[![Download BoneFM.pth](https://img.shields.io/badge/Download-BoneFM.pth-2ea44f?style=flat-square&logo=googledrive&logoColor=white)](https://drive.google.com/uc?id=1NsiBZOx7vAYiN0IDdjYdqFkfArrW_Scn)

Download the released pretrained backbone:

```sh
pip install gdown
gdown "1NsiBZOx7vAYiN0IDdjYdqFkfArrW_Scn" -O BoneFM.pth
```

Place it in the default location:

```sh
mkdir -p finetune/checkpoints
cp BoneFM.pth finetune/checkpoints/
```

Expected path:

```text
finetune/checkpoints/BoneFM.pth
```

## Inference Notebooks

General inference template:

- [BoneCoT_inference.ipynb](BoneCoT_inference.ipynb)

Task-specific 5-fold notebook templates:

- [scripts/Task1_bone_lesion_inference.ipynb](scripts/Task1_bone_lesion_inference.ipynb)
- [scripts/Task2_benign_or_malignant_inference.ipynb](scripts/Task2_benign_or_malignant_inference.ipynb)
- [scripts/Task3_primary_or_metastatic_inference.ipynb](scripts/Task3_primary_or_metastatic_inference.ipynb)

These notebooks are templates only. They assume that you provide:

- A local CSV file describing your private dataset
- The released `BoneFM.pth` backbone checkpoint
- Your own task-specific fine-tuned checkpoints when required

Launch Jupyter:

```sh
cd /path/to/BoneCoT
conda activate bonecot
jupyter notebook
```

Open the desired notebook and update the private local paths in the configuration cell before running the remaining cells.

## License

This repository is released under [CC BY-NC 4.0](LICENSE).
