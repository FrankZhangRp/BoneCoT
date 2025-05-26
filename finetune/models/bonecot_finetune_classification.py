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

        

class BoneCoT_TwoLinearClassification(nn.Module):
    def __init__(self, backbone_model, extra_token_num=2,  embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__()
        autocast_ctx = partial(torch.autocast, enabled=True, dtype=torch.half, device_type="cuda")
        self.num_classes = num_classes
        self.backbone = ModelWithIntermediateLayers_FT(backbone_model, use_n_blocks, autocast_ctx)
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.extra_token_num = extra_token_num
        self.embed_dim = embed_dim
        self.output_dim = self.embed_dim * use_n_blocks + int(use_avgpool) * embed_dim
        self.output_dim = int(self.output_dim)
        self.hidden_dim = self.embed_dim

        self.linear_classifier_1 = nn.Linear(self.output_dim + extra_token_num * self.embed_dim, self.hidden_dim)
        self.extra_token_linear_layers = nn.ModuleList([
            nn.Linear(self.output_dim, self.embed_dim) for _ in range(extra_token_num)
        ])
            
        self.linear_classifier_2 = nn.Linear(self.hidden_dim, self.num_classes)
        
        nn.init.xavier_uniform_(self.linear_classifier_1.weight)
        nn.init.zeros_(self.linear_classifier_1.bias)
        nn.init.xavier_uniform_(self.linear_classifier_2.weight)
        nn.init.zeros_(self.linear_classifier_2.bias)
        self.layernorm = nn.LayerNorm(self.hidden_dim) 
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.GELU()
    
    def init_extra_token_linear_layers(self, model_dict_list):
        if len(model_dict_list) != self.extra_token_num:
            raise ValueError
        for i in range(self.extra_token_num):
            model_dict = model_dict_list[i]
            weight_keys = [k for k in model_dict.keys() if k.endswith('linear_classifier_1.weight') or k.endswith('linear_classifier_1.linear.weight')]
            bias_keys = [k for k in model_dict.keys() if k.endswith('linear_classifier_1.bias') or k.endswith('linear_classifier_1.linear.bias')]
            if not weight_keys or not bias_keys:
                raise ValueError(f"Cannot find weight or bias keys in model_dict for extra token {i}")
                
            weight_key = weight_keys[0]
            bias_key = bias_keys[0]
            self.extra_token_linear_layers[i].weight.data.copy_(model_dict[weight_key])
            self.extra_token_linear_layers[i].bias.data.copy_(model_dict[bias_key])
    
    def forward(self, images, hidden_output=False):
        with torch.no_grad():
            img_features = self.backbone(images)
        img_features = create_linear_input(img_features, self.use_n_blocks, self.use_avgpool)
        extra_token_features = []
        for i in range(self.extra_token_num):
            with torch.no_grad():
                single_extra_token_features = self.extra_token_linear_layers[i](img_features)
            single_extra_token_features = self.layernorm(single_extra_token_features)
            single_extra_token_features = self.activation(single_extra_token_features)
            extra_token_features.append(single_extra_token_features)
        extra_token_features = torch.cat(extra_token_features, dim=-1)
        
        hidden = self.linear_classifier_1(torch.cat([img_features, extra_token_features], dim=-1))
        hidden = self.layernorm(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        pred = self.linear_classifier_2(hidden)
        if hidden_output:
            return pred, hidden
        else:
            return pred
    
    def train(self, mode=True):
        self.backbone.train(mode)
        self.backbone.feature_model.train(mode)
        for i in range(self.extra_token_num):
            self.extra_token_linear_layers[i].train(mode)
        self.linear_classifier_1.train(mode)
        self.linear_classifier_2.train(mode)
        
    def eval(self):
        self.train(mode=False)
    
    def get_optimizer_param_groups(self, backbone_lr, lr):
        param_groups = [
            {"params": self.backbone.feature_model.parameters(), "lr": backbone_lr},
            {"params": self.linear_classifier_1.parameters(), "lr": lr},
            {"params": self.linear_classifier_2.parameters(), "lr": lr},
        ]
        return param_groups


class BoneCoT_MultiRound_InferenceClassification(BoneCoT_TwoLinearClassification):
    def __init__(self, backbone_model, extra_token_num=2,  embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__(backbone_model=backbone_model, extra_token_num=extra_token_num, embed_dim=embed_dim, use_n_blocks=use_n_blocks, use_avgpool=use_avgpool, num_classes=num_classes)
        autocast_ctx = partial(torch.autocast, enabled=True, dtype=torch.half, device_type="cuda")
        self.num_classes = num_classes
        self.backbone = ModelWithIntermediateLayers_FT(backbone_model, use_n_blocks, autocast_ctx)
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.extra_token_num = extra_token_num
        self.embed_dim = embed_dim
            
        self.output_dim = self.embed_dim * use_n_blocks + int(use_avgpool) * embed_dim
        self.output_dim = int(self.output_dim)
        self.hidden_dim = self.embed_dim

        self.linear_classifier_1 = nn.Linear(self.output_dim + extra_token_num * self.embed_dim, self.hidden_dim)

        self.linear_classifier_2 = nn.Linear(self.hidden_dim, self.num_classes)
        
        nn.init.xavier_uniform_(self.linear_classifier_1.weight)
        nn.init.zeros_(self.linear_classifier_1.bias)
        nn.init.xavier_uniform_(self.linear_classifier_2.weight)
        nn.init.zeros_(self.linear_classifier_2.bias)
        self.layernorm = nn.LayerNorm(self.hidden_dim) 
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.GELU()
    
    def init_bonefm_extra_token_linear_layers(self, model_dict_list):
        if len(model_dict_list) != self.extra_token_num:
            raise ValueError
        for i in range(self.extra_token_num):
            model_dict = model_dict_list[i]
            weight_keys = [k for k in model_dict.keys() if k.endswith('linear_classifier_1.weight') or k.endswith('linear_classifier_1.linear.weight')]
            bias_keys = [k for k in model_dict.keys() if k.endswith('linear_classifier_1.bias') or k.endswith('linear_classifier_1.linear.bias')]
            if not weight_keys or not bias_keys:
                raise ValueError(f"Cannot find weight or bias keys in model_dict for extra token {i}")
                
            weight_key = weight_keys[0]
            bias_key = bias_keys[0]
            self.boneform_extra_token_linear_layers[i].weight.data.copy_(model_dict[weight_key])
            self.boneform_extra_token_linear_layers[i].bias.data.copy_(model_dict[bias_key])
            
    def forward(self, images, extra_hidden_features=None, hidden_output=False):
        with torch.no_grad():
            img_features = self.backbone(images)
        img_features = create_linear_input(img_features, self.use_n_blocks, self.use_avgpool)
        if extra_hidden_features is not None:
            extra_token_features = extra_hidden_features
        else:
            extra_token_features = torch.zeros_like(self.embed_dim * self.extra_token_num).to(images.device)
        hidden = self.linear_classifier_1(torch.cat([img_features, extra_token_features], dim=-1))
        hidden = self.layernorm(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        pred = self.linear_classifier_2(hidden)
        if hidden_output:
            return pred, hidden
        else:
            return pred
    
    def train(self, mode=True):
        self.backbone.train(mode)
        self.backbone.feature_model.train(mode)
        self.linear_classifier_1.train(mode)
        self.linear_classifier_2.train(mode)
        
    def eval(self):
        self.train(mode=False)
    
    def get_optimizer_param_groups(self, backbone_lr, lr):
        param_groups = [
            {"params": self.backbone.feature_model.parameters(), "lr": backbone_lr},
            {"params": self.linear_classifier_1.parameters(), "lr": lr},
            {"params": self.linear_classifier_2.parameters(), "lr": lr},
        ]
        return param_groups

class SingleBoneCoT_TwoLinearClassification(BoneCoT_TwoLinearClassification):
    def __init__(self, backbone_model, extra_token_num=2, embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__(backbone_model=backbone_model, extra_token_num=extra_token_num, embed_dim=embed_dim, use_n_blocks=use_n_blocks, use_avgpool=use_avgpool, num_classes=num_classes)
        for param in self.backbone.feature_model.parameters():
            param.requires_grad = False

    def forward(self, images, hidden_output=False):
        with torch.no_grad():
            img_features = self.backbone(images)
        img_features = create_linear_input(img_features, self.use_n_blocks, self.use_avgpool)
        extra_token_features = []
        if self.extra_token_num > 0:
            for i in range(self.extra_token_num):
                with torch.no_grad():
                    single_extra_token_features = self.extra_token_linear_layers[i](img_features)
                single_extra_token_features = self.layernorm(single_extra_token_features)
                single_extra_token_features = self.activation(single_extra_token_features)
                extra_token_features.append(single_extra_token_features)
            extra_token_features = torch.cat(extra_token_features, dim=-1)
            
        if self.extra_token_num > 0:
            hidden = self.linear_classifier_1(torch.cat([img_features, extra_token_features], dim=-1))
        else:
            hidden = self.linear_classifier_1(img_features)
        hidden = self.layernorm(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        pred = self.linear_classifier_2(hidden)
        if hidden_output:
            return pred, hidden
        else:
            return pred
        
    def get_optimizer_param_groups(self, backbone_lr, lr):
        param_groups = [
            {"params": self.linear_classifier_1.parameters(), "lr": lr},
            {"params": self.linear_classifier_2.parameters(), "lr": lr},
        ]
        return param_groups