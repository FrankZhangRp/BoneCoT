from . import vision_transformer as vits
from . import bonefm_finetune_classification as bonefm_classifier
from . import bonecot_finetune_classification as bonecot_classifier
import torch

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

def build_bonefm_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    if torch.cuda.is_available():
        backbone_model = backbone_model.cuda()
    else:
        raise RuntimeError("CUDA is not available.")
    model = bonefm_classifier.BoneFM_Finetune_Classification(backbone_model, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_bonefm_single_linear_finetune_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    if torch.cuda.is_available():
        backbone_model = backbone_model.cuda()
    else:
        raise RuntimeError("CUDA is not available.")
    model = bonefm_classifier.BoneFM_Finetune_Single_Linear_Classification(backbone_model, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_bonecot_linear_finetune_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    model = bonecot_classifier.LinearFinetuneClassificationWithBoneCoT(backbone_model, cfg.data.bonecot_extra_feature_dim, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_bonecot_single_linear_finetune_model_from_cfg(cfg, only_teacher = True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    model = bonecot_classifier.BoneFM_Finetune_Single_Linear_Classification(backbone_model, cfg.data.bonecot_extra_feature_dim, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes)
    return model, embed_dim

def build_only_backbone_feature_model_from_cfg(cfg, only_teacher=True):
    if only_teacher:
        backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    else:
        _, backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    if torch.cuda.is_available():
        backbone_model = backbone_model.cuda()
    else:
        raise RuntimeError("CUDA is not available.")
    model = bonefm_classifier.OnlyBackboneModel(backbone_model, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool)
    return model, embed_dim
