'''
参考Dinov2的官方classifier实现方式
同时训练多个线性分类器 使用不同的结构以及不同的学习率 最后输出最好的结果
'''
import torch
import torch.nn as nn

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
    """单个线性分类器，用于训练冻结特征上的分类器"""

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

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx, default_eval=True):
        super().__init__()
        self.feature_model = feature_model
        if default_eval:
            self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features

class MultiLinearClassifier(nn.Module):
    def __init__(self, sample_output, n_last_blocks_list, learning_rates, num_classes=1000):
        super(MultiLinearClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.n_last_blocks_list = n_last_blocks_list
        self.learning_rates = learning_rates
        self.optim_param_groups = []
        self.linear_classifiers_dict = nn.ModuleDict()

        # 初始化线性分类器
        self.setup_classifiers(sample_output)
        
        self.classifiers_keys = list(self.linear_classifiers_dict.keys())
        
    def unpack_output(self, output):
        """将模型的输出拆解为每个分类器的输入"""
        output_dict = {}
        for i, key in enumerate(self.classifiers_keys):
            output_dict[key] = output[i]
        return output_dict
    
    def setup_classifiers(self, sample_output):
        """初始化所有的线性分类器"""
        for n in self.n_last_blocks_list:
            for avgpool in [False, True]:
                for lr in self.learning_rates:
                    out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                    linear_classifier = LinearClassifier(out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=self.num_classes)
                    self.linear_classifiers_dict[f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")] = linear_classifier
                    self.optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    def forward(self, x_tokens_list):
        """前向传播，返回一个拼接的张量"""
        outputs = [classifier(x_tokens_list) for classifier in self.linear_classifiers_dict.values()]
        # 将所有的分类器输出拼接成一个大张量
        return torch.stack(outputs, dim=0)