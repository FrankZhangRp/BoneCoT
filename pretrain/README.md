# BoneFM Pretraining Implementation Reference

This directory contains the BoneFM pretraining implementation derived from the DINOv2 codebase and adapted for skeleton-focused CT representation learning.

The public documentation for this repository intentionally does not expose internal training recipes, private dataset layouts, non-public reproduction assets, or manuscript-revision analysis paths. The released source is kept here as an implementation reference for readers who need to inspect the model family and code dependencies.

## Public Scope

Included here:

- DINOv2-style model, layer, loss, and utility code used by the BoneFM implementation.
- Configuration files retained for source-code transparency.
- The package metadata needed by local development environments.

Not included here:

- Private pretraining CT data.
- Internal data curation scripts and case manifests.
- Production training launch commands.
- Cluster-specific schedules, paths, and logs.
- Non-public reproduction and release-preparation assets.

## BoneFM Context

BoneFM is the skeleton-focused CT foundation backbone used by BoneCoT. The public release emphasizes downstream use of the released backbone weights, distributed through the Hugging Face model repository:

```sh
hf download frankzhang/BoneFM BoneFM.pth --local-dir ../finetune/checkpoints
```

The expected downstream location is:

```text
finetune/checkpoints/BoneFM.pth
```

For public inference, input CT slices should be prepared with the bone-window convention described in the repository root README:

```text
WL = 300
WW = 1500
```

## DINOv2 Acknowledgement

BoneFM builds on the DINOv2 self-supervised vision-transformer codebase. Please also cite the DINOv2 work when using the inherited implementation:

```bibtex
@misc{oquab2023dinov2,
  title = {DINOv2: Learning Robust Visual Features without Supervision},
  author = {Oquab, Maxime and Darcet, Timothee and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Assran, Mahmoud and Ballas, Nicolas and Galuba, Wojciech and Howes, Russell and Huang, Po-Yao and Li, Shang-Wen and Misra, Ishan and Rabbat, Michael and Sharma, Vasu and Synnaeve, Gabriel and Xu, Hu and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  year = {2023},
  eprint = {2304.07193},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```
