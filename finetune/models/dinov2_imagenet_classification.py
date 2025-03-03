'''
使用ImageNet微调后的Dinov2模型导入 官方并没有公布ImageNet微调前面head的模型
注意后期摒弃联网过程
'''

# Load model directly
# from transformers import AutoImageProcessor, AutoModelForImageClassification

# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant-imagenet1k-1-layer")
# model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-giant-imagenet1k-1-layer")
import torch

dinov2_vitg14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc')
