'''
线性分类器微调
注意参考DINOv2和RayDINO的分类器结构
两种考虑 一种是最简单的 在backbone的特征层后面接上一个线性分类器
另一种是参考only_linear的实现 在每一个block的输出后面都接一个线性分类器
考虑全参数微调的学习率设置差异
'''


import torch
import torch.nn as nn

from .linear_only_classification import create_linear_input
from functools import partial

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

def split_o1_input_tensors(merged_tensor):
    """
    从合并后的 tensor 中拆分出 image 和 feature。
    merged_tensor: [batch_size, 3 + feature_channels, height, width]
    返回 image: [batch_size, 3, height, width] 和 feature: [batch_size, feature_channels]
    """
    # image 是前三个通道
    image = merged_tensor[:, :3, :, :]
    
    # feature 是从第四个通道开始的部分
    feature = merged_tensor[:, 3:, :, :]
    
    # 将 feature 的空间维度还原为 [batch_size, feature_channels]
    feature = feature[:, :, 0, 0]
    
    return image, feature

class LinearClassifierWithExtraFeature(nn.Module):
    """单个线性分类器，用于训练冻结特征上的分类器"""

    def __init__(self, out_dim, extra_feature_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.extra_feature_dim = extra_feature_dim
        self.linear = nn.Linear(out_dim + extra_feature_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list, extra_feature):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(torch.cat([output, extra_feature], dim=-1))

class ModelWithIntermediateLayers_FT(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with self.autocast_ctx():
            features = self.feature_model.get_intermediate_layers(images, self.n_last_blocks, return_class_token=True)
        return features
    
class LinearFinetuneClassificationWithO1(nn.Module):
    def __init__(self, backbone_model, extra_feature_dim=34, embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__()
        autocast_ctx = partial(torch.autocast, enabled=True, dtype=torch.half, device_type="cuda")
        self.num_classes = num_classes
        self.backbone = ModelWithIntermediateLayers_FT(backbone_model, use_n_blocks, autocast_ctx)
        self.embed_dim = embed_dim
        self.output_dim = self.embed_dim * use_n_blocks + int(use_avgpool) * embed_dim
        self.output_dim = int(self.output_dim)
        self.extra_feature_dim = extra_feature_dim
        self.linear_classifier = LinearClassifierWithExtraFeature(self.output_dim, self.extra_feature_dim, use_n_blocks, use_avgpool, self.num_classes)
    def forward(self, input_tensors):
        images, extra_feature = split_o1_input_tensors(input_tensors)
        features = self.backbone(images)
        return self.linear_classifier(features, extra_feature)
    
    def train(self, mode=True):
        self.backbone.train(mode)
        self.backbone.feature_model.train(mode)
        self.linear_classifier.train(mode)
    
    def eval(self):
        self.train(mode=False)
    
    def get_optimizer_param_groups(self, backbone_lr, lr):
        """
        返回包含backbone和linear_classifier的参数组，分别设置学习率
        """
        param_groups = [
            {"params": self.backbone.feature_model.parameters(), "lr": backbone_lr},
            {"params": self.linear_classifier.parameters(), "lr": lr},
        ]
        return param_groups

class OnlyBackboneModel(nn.Module):
    def __init__(self, backbone_model, embed_dim=1536, use_n_blocks=4, use_avgpool=True):
        super().__init__()
        autocast_ctx = partial(torch.autocast, enabled=True, dtype=torch.half, device_type="cuda")
        self.backbone = ModelWithIntermediateLayers_FT(backbone_model, use_n_blocks, autocast_ctx)
        self.embed_dim = embed_dim
        self.output_dim = self.embed_dim * use_n_blocks + int(use_avgpool) * embed_dim
        self.output_dim = int(self.output_dim)
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        
    def forward(self, images):
        features = self.backbone(images)
        return create_linear_input(features, self.use_n_blocks, self.use_avgpool)
    
    def train(self, mode=True):
        self.backbone.train(mode)
        self.backbone.feature_model.train(mode)
    
    def eval(self):
        self.train(mode=False)
    
    def get_optimizer_param_groups(self, backbone_lr, lr):
        """
        返回包含backbone和linear_classifier的参数组，分别设置学习率
        """
        param_groups = [
            {"params": self.backbone.feature_model.parameters(), "lr": lr},
        ]
        return param_groups
    
        
class SingleLinearFinetuneClassificationWithO1(LinearFinetuneClassificationWithO1):
    def __init__(self, backbone_model,  extra_feature_dim=34, embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__(backbone_model=backbone_model, extra_feature_dim=34, embed_dim=embed_dim, use_n_blocks=use_n_blocks, use_avgpool=use_avgpool, num_classes=num_classes)
        # 把backbone的grads关闭
        for param in self.backbone.feature_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_tensors):
        images, extra_feature = split_o1_input_tensors(input_tensors)
        with torch.no_grad():
            features = self.backbone(images)
        return self.linear_classifier(features, extra_feature)
        
    def get_optimizer_param_groups(self, backbone_lr, lr):
        """
        返回包含backbone和linear_classifier的参数组，分别设置学习率
        """
        param_groups = [
            {"params": self.linear_classifier.parameters(), "lr": lr},
        ]
        return param_groups


