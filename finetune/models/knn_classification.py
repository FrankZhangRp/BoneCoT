'''
参考Dinov2的官方classifier实现方式

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.parallel import DistributedDataParallel
import numpy as np

def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


class KNN_Classifier(nn.Module):
    def __init__(self, backbone, num_classes=10, k=5):
        super(KNN_Classifier, self).__init__()
        self.backbone = backbone
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.num_classes = num_classes
        
        # 设置一个标记来控制训练时是否使用KNN
        self.training_mode = True
        
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        if self.training_mode:
            # 训练阶段只返回特征
            return features
        else:
            # 测试阶段使用KNN分类器
            preds = self.knn.predict(features.detach().cpu().numpy())
            return torch.tensor(preds).to(features.device) # 为了兼容dataparallel的实现

    def fit_knn(self, features, labels):
        # 用提取的特征训练KNN分类器
        # 判断特征是tensor还是numpy
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            labels = labels.cpu().numpy()
        elif isinstance(features, np.ndarray):
            pass
        self.knn.fit(features, labels)
    
    def train(self, mode=True):
        self.backbone.eval()
        self.training_mode = mode
    
    def eval(self):
        self.train(False)