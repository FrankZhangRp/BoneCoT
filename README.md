# BoneCoT

Official code release for "BoneCoT: Multi-center validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought".

This repository contains:

- `BoneFM` pretraining code built on top of DINOv2
- Fine-tuning and evaluation code for BoneFM and BoneCoT
- Public notebook templates for downstream task inference

## Public Release Scope

Included:

- Source code
- Public notebook templates
- The released `BoneFM.pth` pretrained backbone checkpoint

Not included:

- Training, validation, or test datasets
- Task-specific fine-tuned checkpoints
- Review-only demo assets from the anonymous submission version

To run local evaluation, replace the placeholder paths in the notebooks or config files with your own private assets.

## Repository Structure

- `pretrain/`: BoneFM pretraining pipeline based on DINOv2
- `finetune/`: fine-tuning and evaluation code for BoneFM and BoneCoT
- `scripts/`: 5-fold notebook templates for task-specific evaluation
- `BoneCoT_inference.ipynb`: local BoneCoT inference template

## Environment

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

## Released Checkpoint

Download the BoneFM pretrained backbone checkpoint:

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

- [BoneCoT_inference.ipynb](BoneCoT_inference.ipynb): BoneCoT inference template

Task-specific 5-fold notebook templates:

- [scripts/Task1_bone_lesion_inference.ipynb](scripts/Task1_bone_lesion_inference.ipynb)
- [scripts/Task2_benign_or_malignant_inference.ipynb](scripts/Task2_benign_or_malignant_inference.ipynb)
- [scripts/Task3_primary_or_metastatic_inference.ipynb](scripts/Task3_primary_or_metastatic_inference.ipynb)

These notebooks are templates only. They assume that you provide:

- A local CSV file describing your private dataset
- The released `BoneFM.pth` backbone checkpoint
- Your own task-specific fine-tuned checkpoints when required

## Running the Notebooks

```sh
cd /path/to/BoneCoT
conda activate bonecot
jupyter notebook
```

Open the desired notebook and update the private local paths in the configuration cell before running the remaining cells.

## License

This repository is released under [CC BY-NC 4.0](LICENSE).
