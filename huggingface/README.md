---
license: cc-by-nc-4.0
language:
- en
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
- medical-imaging
- computed-tomography
- bone-window
- skeleton
- foundation-model
- vision-transformer
- dinov2
- bone-metastases
- bone-disease
---

# BoneFM

BoneFM is the skeleton-focused CT foundation backbone used by **BoneCoT: multicentre validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought**.

- Paper DOI: [10.1038/s41551-026-01736-1](https://doi.org/10.1038/s41551-026-01736-1)
- Code: [FrankZhangRp/BoneCoT](https://github.com/FrankZhangRp/BoneCoT)
- Project page: [frankzhangrp.github.io/BoneCoT](https://frankzhangrp.github.io/BoneCoT/)

## Model Summary

BoneFM is a Vision Transformer backbone adapted from DINOv2-style self-supervised learning for skeleton-focused CT representation. BoneCoT uses BoneFM features and clinician-derived task dependencies for downstream bone metastasis and bone-related disease reasoning.

This repository hosts the public BoneFM backbone checkpoint:

| File | Description |
| --- | --- |
| `BoneFM.pth` | BoneFM pretrained backbone checkpoint for the BoneCoT public code |
| `README.md` | Hugging Face model card |

Checkpoint integrity:

- Size: `4,946,789,774` bytes
- SHA256: `5bed7f117e4f8a9f3b11eded0408e9ba60ee0bf3c3b335982d6b9e608c69d271`

## Intended Use

BoneFM is intended for non-commercial research on skeletal CT representation learning and downstream bone-related disease modelling. It can be used as a feature backbone with the public BoneCoT code when users provide their own de-identified image data and clinically appropriate labels.

BoneFM and BoneCoT are not standalone clinical diagnostic devices. They should not be used for patient management without local validation, regulatory review, and qualified clinical oversight.

## Input Convention

Prepare CT slices with the bone-window convention used by the public BoneCoT code:

```text
WL = 300
WW = 1500
image = clip((HU - (WL - WW / 2)) / WW, 0, 1)
```

The public code expects PIL-readable RGB-compatible image files and applies the evaluation transforms defined in the BoneCoT repository.

## Download

Download the released checkpoint:

```sh
hf download frankzhang/BoneFM BoneFM.pth --local-dir finetune/checkpoints
```

Python alternative:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="frankzhang/BoneFM",
    filename="BoneFM.pth",
    local_dir="finetune/checkpoints",
)
print(path)
```

The expected local path for the BoneCoT repository is:

```text
finetune/checkpoints/BoneFM.pth
```

## Public Release Boundary

This model repository is for the BoneFM backbone checkpoint and model card. It does not include:

- Private clinical training, validation, or test datasets.
- Patient-level metadata.
- Non-public reproduction packages.
- Internal training launch recipes or cluster-specific paths.
- Task-specific fine-tuned checkpoints unless separately released.

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

BoneFM builds on DINOv2-style self-supervised vision-transformer code. Please also cite the relevant DINOv2 work when using inherited implementation components.

## License

The public BoneFM release is made available under CC BY-NC 4.0 for non-commercial research use, subject to any applicable third-party code licenses in the accompanying implementation.
