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

class LinearClassifier(nn.Module):
    """Linear classifier for training on frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


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


    
class BoneFMTwoLinearFinetuneClassification(nn.Module):
    def __init__(self, backbone_model, embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__()
        autocast_ctx = partial(torch.autocast, enabled=True, dtype=torch.half, device_type="cuda")
        self.num_classes = num_classes
        self.backbone = ModelWithIntermediateLayers_FT(backbone_model, use_n_blocks, autocast_ctx)
        self.embed_dim = embed_dim
        self.output_dim = self.embed_dim * use_n_blocks + int(use_avgpool) * embed_dim
        self.output_dim = int(self.output_dim)
        self.hidden_dim = self.embed_dim  # hidden_dim和embed_dim相同 一个token的维度
        self.linear_classifier_1 = LinearClassifier(self.output_dim, use_n_blocks, use_avgpool, self.hidden_dim)
        self.linear_classifier_2 = nn.Linear(self.hidden_dim, self.num_classes)
        
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.linear_classifier_1.linear.weight)
        nn.init.zeros_(self.linear_classifier_1.linear.bias)
        nn.init.xavier_uniform_(self.linear_classifier_2.weight)
        nn.init.zeros_(self.linear_classifier_2.bias)
        self.layernorm = nn.LayerNorm(self.hidden_dim) 
        self.dropout = nn.Dropout(0.3)  # 添加dropout层
        self.activation = nn.GELU()  # 使用GELU作为更平滑的激活函数
        
    def forward(self, images, hidden_output=False):
        features = self.backbone(images)
        hidden = self.linear_classifier_1(features)
        hidden = self.layernorm(hidden)
        # 使用GELU激活函数
        hidden = self.activation(hidden)
        # 添加dropout
        hidden = self.dropout(hidden)
        # 线性层
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
        """
        返回包含backbone和linear_classifier的参数组，分别设置学习率
        """
        param_groups = [
            {"params": self.backbone.feature_model.parameters(), "lr": backbone_lr},
            {"params": self.linear_classifier_1.parameters(), "lr": lr},
            {"params": self.linear_classifier_2.parameters(), "lr": lr},
        ]
        return param_groups
    
class BoneFM_Classification(BoneFMTwoLinearFinetuneClassification):
    def __init__(self, backbone_model, embed_dim=1536, use_n_blocks=4, use_avgpool=True,  num_classes=100):
        super().__init__(backbone_model=backbone_model, embed_dim=embed_dim, use_n_blocks=use_n_blocks, use_avgpool=use_avgpool, num_classes=num_classes)
        # 把backbone的grads关闭
        for param in self.backbone.feature_model.parameters():
            param.requires_grad = False

    def forward(self, images, hidden_output=False):
        features = self.backbone(images)
        hidden = self.linear_classifier_1(features)
        hidden = self.layernorm(hidden)
        # 使用GELU激活函数
        hidden = self.activation(hidden)
        # 添加dropout
        hidden = self.dropout(hidden)
        # 线性层
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


