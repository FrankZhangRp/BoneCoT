import torch
import torch.nn as nn
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

def split_bonecot_input_tensors(merged_tensor):
    # First 3 channels are image
    image = merged_tensor[:, :3, :, :]
    
    # Remaining channels are features
    feature = merged_tensor[:, 3:, :, :]
    
    # Reshape feature to [batch_size, feature_channels]
    feature = feature[:, :, 0, 0]
    
    return image, feature

class LinearClassifierWithExtraFeature(nn.Module):
    """Linear classifier for training on frozen features"""

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
    
class LinearFinetuneClassificationWithBoneCoT(nn.Module):
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
        images, extra_feature = split_bonecot_input_tensors(input_tensors)
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
        Return parameter groups for backbone and linear_classifier with different learning rates
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
        Return parameter groups for backbone and linear_classifier with different learning rates
        """
        param_groups = [
            {"params": self.backbone.feature_model.parameters(), "lr": lr},
        ]
        return param_groups
    
        
class SingleLinearFinetuneClassificationWithBoneCoT(LinearFinetuneClassificationWithBoneCoT):
    def __init__(self, backbone_model,  extra_feature_dim=34, embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__(backbone_model=backbone_model, extra_feature_dim=34, embed_dim=embed_dim, use_n_blocks=use_n_blocks, use_avgpool=use_avgpool, num_classes=num_classes)
        # Disable gradients for backbone
        for param in self.backbone.feature_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_tensors):
        images, extra_feature = split_bonecot_input_tensors(input_tensors)
        with torch.no_grad():
            features = self.backbone(images)
        return self.linear_classifier(features, extra_feature)
        
    def get_optimizer_param_groups(self, backbone_lr, lr):
        """
        Return parameter groups for backbone and linear_classifier with different learning rates
        """
        param_groups = [
            {"params": self.linear_classifier.parameters(), "lr": lr},
        ]
        return param_groups

