# BoneCoT

BoneCoT: multicentre validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought.

<p align="center">
  <a href="https://frankzhangrp.github.io/BoneCoT/"><img src="https://img.shields.io/badge/Project-Page-0f766e?style=flat-square" alt="Project Page"></a>
  <a href="https://doi.org/10.1038/s41551-026-01736-1"><img src="https://img.shields.io/badge/DOI-10.1038%2Fs41551--026--01736--1-blue?style=flat-square" alt="DOI"></a>
  <a href="https://huggingface.co/frankzhang/BoneFM"><img src="https://img.shields.io/badge/Hugging%20Face-BoneFM-ffcc4d?style=flat-square" alt="Hugging Face model"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
</p>

Official code release for the BoneCoT paper in *Nature Biomedical Engineering*.

## News

- The paper is scheduled for online publication in *Nature Biomedical Engineering* on **02 July 2026, 10:00 London time**. DOI: [10.1038/s41551-026-01736-1](https://doi.org/10.1038/s41551-026-01736-1).
- The BoneFM backbone weights are released through [Hugging Face](https://huggingface.co/frankzhang/BoneFM), instead of being stored directly in this GitHub repository.
- A GitHub Pages project page is available under [`docs/`](docs/) for the paper summary, model overview, and release links.

## Overview

BoneCoT is a clinician-guided chain-of-thought framework for bone metastasis and bone-related disease assessment from whole-body skeletal CT. The model is built on **BoneFM**, a skeleton-focused CT foundation backbone adapted from DINOv2-style self-supervised pretraining, and then uses clinician-derived task dependencies to support downstream bone lesion reasoning.

This public repository focuses on reusable code, inference templates, model-loading instructions, and release documentation. Private clinical datasets, internal training runs, non-public reproduction packages, and manuscript-revision analysis materials are not surfaced in the public README.

## What This Repository Provides

### Included

- BoneFM/BoneCoT model definitions and evaluation utilities.
- Public inference notebook templates for local data.
- Configuration templates with placeholder paths for private user assets.
- A Hugging Face model card under [`huggingface/README.md`](huggingface/README.md).
- A GitHub Pages project page under [`docs/`](docs/).

### Not Included

- Training, validation, or test datasets.
- Private clinical annotations or patient-level metadata.
- Task-specific fine-tuned checkpoints unless separately released.
- Non-public reproduction packages and release-preparation assets.
- Internal training recipes, ablation scripts, and manuscript-revision analysis pipelines.

## CT Input Convention

BoneFM and BoneCoT operate on bone-window CT image slices. For local inference, prepare input images from CT HU values using the following public preprocessing convention:

| Item | Value |
| --- | --- |
| Window level | `300` |
| Window width | `1500` |
| Intensity mapping | `clip((HU - (WL - WW / 2)) / WW, 0, 1)` |
| Image format | 2D RGB-compatible image files loaded by PIL |
| Default crop size | `518` |
| Normalization | ImageNet mean/std in the released evaluation transforms |

The repository expects already prepared image files in the CSV manifests. Raw DICOM/NIfTI conversion, private de-identification, and cohort curation are intentionally outside the public release.

## Repository Structure

| Path | Description |
| --- | --- |
| `finetune/` | BoneFM and BoneCoT model definitions, evaluators, configs, and metrics |
| `scripts/` | Task-specific inference notebook templates |
| `BoneCoT_inference.ipynb` | General local inference template |
| `pretrain/` | BoneFM/DINOv2 implementation reference; public README omits internal training commands |
| `huggingface/` | Model-card README for the Hugging Face BoneFM repository |
| `docs/` | Static GitHub Pages project page |

## Quick Start

### Environment

Recommended setup:

- OS: Ubuntu 22.04
- GPU: NVIDIA GPU with at least 24 GB VRAM
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

## BoneFM Backbone Weights

The recommended public distribution path is Hugging Face:

```sh
hf download frankzhang/BoneFM BoneFM.pth --local-dir finetune/checkpoints
```

Expected local path:

```text
finetune/checkpoints/BoneFM.pth
```

Checkpoint integrity:

- Size: `4,946,789,774` bytes
- SHA256: `5bed7f117e4f8a9f3b11eded0408e9ba60ee0bf3c3b335982d6b9e608c69d271`

Python alternative:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="frankzhang/BoneFM",
    filename="BoneFM.pth",
    local_dir="finetune/checkpoints",
)
```

GitHub is kept code-focused. Large model files should remain on Hugging Face or another model registry rather than being committed to this repository.

## Inference Templates

General inference template:

- [BoneCoT_inference.ipynb](BoneCoT_inference.ipynb)

Task-specific templates:

- [scripts/Task1_bone_lesion_inference.ipynb](scripts/Task1_bone_lesion_inference.ipynb)
- [scripts/Task2_benign_or_malignant_inference.ipynb](scripts/Task2_benign_or_malignant_inference.ipynb)
- [scripts/Task3_primary_or_metastatic_inference.ipynb](scripts/Task3_primary_or_metastatic_inference.ipynb)

These notebooks are templates only. They assume that you provide:

- A CSV manifest for your local, de-identified image files.
- The released `BoneFM.pth` backbone checkpoint.
- Task-specific classifier checkpoints if you are running a task that requires them.

Launch Jupyter:

```sh
conda activate bonecot
jupyter notebook
```

Open the desired notebook and update the private local paths in the configuration cell before running the remaining cells.

## Clinical and Research Use

This repository is intended for research use. BoneCoT is not a standalone clinical diagnostic device, and model outputs should not be used for patient management without local validation, regulatory review, and qualified clinical oversight.

## Citation

Please cite the final *Nature Biomedical Engineering* record once it is live:

```bibtex
@article{bonecot2026,
  title = {BoneCoT: multicentre validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought},
  journal = {Nature Biomedical Engineering},
  year = {2026},
  doi = {10.1038/s41551-026-01736-1}
}
```

## License

This repository is released under [CC BY-NC 4.0](LICENSE).
