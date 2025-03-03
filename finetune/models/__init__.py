# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from . import vision_transformer as vits
from . import knn_classification as knn_classifier
from . import linear_only_classification as lo_classifier
from . import linear_finetune_classification as lf_classifier
from . import ctvit_base as ctvit
from . import o1_linear_finetune_classification as o1_classifier
import torch
from functools import partial
def build_backbone_model(args, only_teacher=True, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_backbone_model_from_cfg(cfg, only_teacher=True):
    return build_backbone_model(cfg.model, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)


def build_knn_model_from_cfg(cfg, only_teacher=True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    knn_model = knn_classifier.KNN_Classifier(backbone_model, cfg.model.num_classes, cfg.model.knn_k)
    return knn_model, embed_dim

def build_only_linear_model_from_cfg(cfg, only_teacher=True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    if torch.cuda.is_available():
        backbone_model = backbone_model.cuda()
    else:
        raise RuntimeError("CUDA is not available.")
    n_last_blocks = max(cfg.model.n_last_blocks_list)
    autocast_ctx = partial(torch.autocast, enabled=True, dtype=torch.half, device_type="cuda")
    feature_model = lo_classifier.ModelWithIntermediateLayers(backbone_model, n_last_blocks, autocast_ctx)
    sample_input = torch.randn(1, 3, cfg.crops.global_crops_size, cfg.crops.global_crops_size).cuda()
    sample_output = feature_model(sample_input)
    linear_model = lo_classifier.MultiLinearClassifier(sample_output, cfg.model.n_last_blocks_list, cfg.optim.learning_rates, cfg.model.num_classes)
    return backbone_model, feature_model, linear_model, embed_dim

def build_linear_finetune_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    if torch.cuda.is_available():
        backbone_model = backbone_model.cuda()
    else:
        raise RuntimeError("CUDA is not available.")
    model = lf_classifier.LinearFinetuneClassification(backbone_model, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_single_linear_finetune_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    if torch.cuda.is_available():
        backbone_model = backbone_model.cuda()
    else:
        raise RuntimeError("CUDA is not available.")
    model = lf_classifier.SingleLinearFinetuneClassification(backbone_model, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_o1_linear_finetune_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    model = o1_classifier.LinearFinetuneClassificationWithO1(backbone_model, cfg.data.o1_extra_feature_dim, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_o1_single_linear_finetune_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    model = o1_classifier.SingleLinearFinetuneClassificationWithO1(backbone_model, cfg.data.o1_extra_feature_dim, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_ct_clip_finetune_model_from_cfg(cfg):
    visual_backbone_model = ctvit.CTViT(dim = 512, codebook_size = 8192, image_size = 480, patch_size = 30, temporal_patch_size = 15, spatial_depth = 4, temporal_depth = 4, dim_head = 32, heads = 8)
    model = ctvit.ImageLatentsClassifier(visual_backbone_model, 512, cfg.model.num_classes)
    return model, 512

def build_only_backbone_feature_model_from_cfg(cfg, only_teacher=True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    if torch.cuda.is_available():
        backbone_model = backbone_model.cuda()
    else:
        raise RuntimeError("CUDA is not available.")
    model = lf_classifier.OnlyBackboneModel(backbone_model, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool)
    return model, embed_dim
