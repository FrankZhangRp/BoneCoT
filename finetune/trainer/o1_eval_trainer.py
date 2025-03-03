from .base_trainer import BaseTrainer
from models import build_o1_linear_finetune_model_from_cfg, build_o1_single_linear_finetune_model_from_cfg
import torch
import traceback
from torchvision import transforms
from data import O1ClassificationDataset
from tqdm import tqdm
import os
import numpy as np

o1_checkpoint_dict = {
    'bone_lesion': [
     '/path/to/checkpoints/bone_lesion/fold_0.pth',
     '/path/to/checkpoints/bone_lesion/fold_1.pth',
     '/path/to/checkpoints/bone_lesion/fold_2.pth',
     '/path/to/checkpoints/bone_lesion/fold_3.pth',
     '/path/to/checkpoints/bone_lesion/fold_4.pth',
    ],
    'benign_malignant': [
    '/path/to/checkpoints/benign_malignant/fold_0.pth',
    '/path/to/checkpoints/benign_malignant/fold_1.pth',
    '/path/to/checkpoints/benign_malignant/fold_2.pth',
    '/path/to/checkpoints/benign_malignant/fold_3.pth',
    '/path/to/checkpoints/benign_malignant/fold_4.pth',
    ],
    'primary_metastasis':[
    '/path/to/checkpoints/primary_metastasis/fold_0.pth',
    '/path/to/checkpoints/primary_metastasis/fold_1.pth',
    '/path/to/checkpoints/primary_metastasis/fold_2.pth',
    '/path/to/checkpoints/primary_metastasis/fold_3.pth',
    '/path/to/checkpoints/primary_metastasis/fold_4.pth',
    ],
    'osteoblastic':[
    '/path/to/checkpoints/chenggu/fold_0.pth',
    '/path/to/checkpoints/chenggu/fold_1.pth',
    '/path/to/checkpoints/chenggu/fold_2.pth',
    '/path/to/checkpoints/chenggu/fold_3.pth',
    '/path/to/checkpoints/chenggu/fold_4.pth',
    ],
    'osteolytic':[
    '/path/to/checkpoints/ronggu/fold_0.pth',
    '/path/to/checkpoints/ronggu/fold_1.pth',
    '/path/to/checkpoints/ronggu/fold_2.pth',
    '/path/to/checkpoints/ronggu/fold_3.pth',
    '/path/to/checkpoints/ronggu/fold_4.pth',
    ],
    'spinal_cord_compression':[
    '/path/to/checkpoints/spinal_cord_compression/fold_0.pth',
    '/path/to/checkpoints/spinal_cord_compression/fold_1.pth',
    '/path/to/checkpoints/spinal_cord_compression/fold_2.pth',
    '/path/to/checkpoints/spinal_cord_compression/fold_3.pth',
    '/path/to/checkpoints/spinal_cord_compression/fold_4.pth',
    ],
    'thrombosis':[
    '/path/to/checkpoints/thrombosis/fold_0.pth',
    '/path/to/checkpoints/thrombosis/fold_1.pth',
    '/path/to/checkpoints/thrombosis/fold_2.pth',
    '/path/to/checkpoints/thrombosis/fold_3.pth',
    '/path/to/checkpoints/thrombosis/fold_4.pth',
    ],
    'pathological_fracture':[
    '/path/to/checkpoints/pathological_fracture/fold_0.pth',
    '/path/to/checkpoints/pathological_fracture/fold_1.pth',
    '/path/to/checkpoints/pathological_fracture/fold_2.pth',
    '/path/to/checkpoints/pathological_fracture/fold_3.pth',
    '/path/to/checkpoints/pathological_fracture/fold_4.pth',
    ],
    'renal_insufficiency':[
    '/path/to/checkpoints/renal_insufficiency/fold_0.pth',
    '/path/to/checkpoints/renal_insufficiency/fold_1.pth',
    '/path/to/checkpoints/renal_insufficiency/fold_2.pth',
    '/path/to/checkpoints/renal_insufficiency/fold_3.pth',
    '/path/to/checkpoints/renal_insufficiency/fold_4.pth',
    ],
    'hypercalcemia':[
    '/path/to/checkpoints/hypercalcemia/fold_0.pth',
    '/path/to/checkpoints/hypercalcemia/fold_1.pth',
    '/path/to/checkpoints/hypercalcemia/fold_2.pth',
    '/path/to/checkpoints/hypercalcemia/fold_3.pth',
    '/path/to/checkpoints/hypercalcemia/fold_4.pth',
    ],
    'type_of_primary_tumor':[
    '/path/to/checkpoints/type_of_primary_tumor/fold_0.pth',
    '/path/to/checkpoints/type_of_primary_tumor/fold_1.pth',
    '/path/to/checkpoints/type_of_primary_tumor/fold_2.pth',
    '/path/to/checkpoints/type_of_primary_tumor/fold_3.pth',
    '/path/to/checkpoints/type_of_primary_tumor/fold_4.pth',
    ],
    'lung_SCLC_or_NSCLC':[
    '/path/to/checkpoints/lung_SCLC_or_NSCLC/fold_0.pth',
    '/path/to/checkpoints/lung_SCLC_or_NSCLC/fold_1.pth',
    '/path/to/checkpoints/lung_SCLC_or_NSCLC/fold_2.pth',
    '/path/to/checkpoints/lung_SCLC_or_NSCLC/fold_3.pth',
    '/path/to/checkpoints/lung_SCLC_or_NSCLC/fold_4.pth',
    ],
    'lung_EGFR':[
    '/path/to/checkpoints/lung_EGFR/fold_0.pth',
    '/path/to/checkpoints/lung_EGFR/fold_1.pth',
    '/path/to/checkpoints/lung_EGFR/fold_2.pth',
    '/path/to/checkpoints/lung_EGFR/fold_3.pth',
    '/path/to/checkpoints/lung_EGFR/fold_4.pth',
    ],
    'lung_ROS1':[
    '/path/to/checkpoints/lung_ROS1/fold_0.pth',
    '/path/to/checkpoints/lung_ROS1/fold_1.pth',
    '/path/to/checkpoints/lung_ROS1/fold_2.pth',
    '/path/to/checkpoints/lung_ROS1/fold_3.pth',
    '/path/to/checkpoints/lung_ROS1/fold_4.pth',
    ],
    'lung_ALK':[
    '/path/to/checkpoints/lung_ALK/fold_0.pth',
    '/path/to/checkpoints/lung_ALK/fold_1.pth',
    '/path/to/checkpoints/lung_ALK/fold_2.pth',
    '/path/to/checkpoints/lung_ALK/fold_3.pth',
    '/path/to/checkpoints/lung_ALK/fold_4.pth',
    ],
    'breast_ER':[
    '/path/to/checkpoints/breast_ER/fold_0.pth',
    '/path/to/checkpoints/breast_ER/fold_1.pth',
    '/path/to/checkpoints/breast_ER/fold_2.pth',
    '/path/to/checkpoints/breast_ER/fold_3.pth',
    '/path/to/checkpoints/breast_ER/fold_4.pth',
    ],
    'breast_PR':[
    '/path/to/checkpoints/breast_PR/fold_0.pth',
    '/path/to/checkpoints/breast_PR/fold_1.pth',
    '/path/to/checkpoints/breast_PR/fold_2.pth',
    '/path/to/checkpoints/breast_PR/fold_3.pth',
    '/path/to/checkpoints/breast_PR/fold_4.pth',
    ],
    'breast_HER2':[
    '/path/to/checkpoints/breast_HER2/fold_0.pth',
    '/path/to/checkpoints/breast_HER2/fold_1.pth',
    '/path/to/checkpoints/breast_HER2/fold_2.pth',
    '/path/to/checkpoints/breast_HER2/fold_3.pth',
    '/path/to/checkpoints/breast_HER2/fold_4.pth',
    ],
    'breast_AR':[
    '/path/to/checkpoints/breast_AR/fold_0.pth',
    '/path/to/checkpoints/breast_AR/fold_1.pth',
    '/path/to/checkpoints/breast_AR/fold_2.pth',
    '/path/to/checkpoints/breast_AR/fold_3.pth',
    '/path/to/checkpoints/breast_AR/fold_4.pth',
    ],
    
}


def merge_o1_input_tensors(image, feature):
    _, _, height, width = image.shape
    expanded_feature = feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
    combined_tensor = torch.cat([image, expanded_feature], dim=1)
    return combined_tensor

class O1EvalTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def initlize(self):
        self.get_logger()
        self.get_criterion()
        self.get_data()
        self.get_model()
        self.load_checkpoint()
        self.get_metric()
        self.setup_model()

    def get_data(self): 
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.args.crops.global_crops_size, scale=(self.args.crops.global_crops_scale[0], self.args.crops.global_crops_scale[1])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=[90, 270]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(self.args.crops.global_crops_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        assert self.args.data.train_dataset is not None
        self.train_dataset = O1ClassificationDataset(self.args.data.train_dataset, transform=self.train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        assert self.args.data.val_dataset is not None
        self.val_dataset = O1ClassificationDataset(self.args.data.val_dataset, transform=self.test_transform)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)
        if self.args.data.test_dataset is not None:
            self.test_dataset = O1ClassificationDataset(self.args.data.test_dataset, transform=self.test_transform)
        else:
            self.test_dataset = self.val_dataset
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)
        
        self.datasets_dict = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        
        self.dataloaders_dict = {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
        
        self.logger.info(f"Train dataset: {self.args.data.train_dataset} / Length: {len(self.train_dataset)} / loader: {len(self.train_dataloader)}")
        self.logger.info(f"Val dataset: {self.args.data.val_dataset} / Length: {len(self.val_dataset)} / loader: {len(self.val_dataloader)}")
        self.logger.info(f"Test dataset: {self.args.data.test_dataset} / Length: {len(self.test_dataset)} / loader: {len(self.test_dataloader)}")
    
        
    
    def get_model(self):
        self.args.data.o1_extra_feature_dim = 16
        self.args.model.num_classes = 1
        self.o1_bone_lesion_feature_dim = 16
        self.o1_bone_lesion_num_classes = 1
        self.model_bone_lesion, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model bone lesion Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_benign_malignant_feature_dim = 15
        self.o1_benign_malignant_num_classes = 1
        self.model_benign_malignant, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model benign malignant Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_primary_metastasis_feature_dim = 15
        self.o1_primary_metastasis_num_classes = 1
        self.model_primary_metastasis, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model primary metastasis Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_osteoblastic_feature_dim = 15
        self.o1_osteoblastic_num_classes = 1
        self.model_osteoblastic, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model osteoblastic Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_osteolytic_feature_dim = 15
        self.o1_osteolytic_num_classes = 1
        self.model_osteolytic, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model osteolytic Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_spinal_cord_compression_feature_dim = 15
        self.o1_spinal_cord_compression_num_classes = 1
        self.model_spinal_cord_compression, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model spinal cord compression Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_thrombosis_feature_dim = 15
        self.o1_thrombosis_num_classes = 1
        self.model_thrombosis, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model thrombosis Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_pathological_fracture_feature_dim = 15
        self.o1_pathological_fracture_num_classes = 1
        self.model_pathological_fracture, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model pathological fracture Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_renal_insufficiency_feature_dim = 15
        self.o1_renal_insufficiency_num_classes = 1
        self.model_renal_insufficiency, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model renal insufficiency Finished")
        
        self.args.data.o1_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.o1_hypercalcemia_feature_dim = 15
        self.o1_hypercalcemia_num_classes = 1
        self.model_hypercalcemia, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model hypercalcemia Finished")
        
        
        self.args.data.o1_extra_feature_dim = 8
        self.args.model.num_classes = 8
        self.o1_type_of_primary_tumor_feature_dim = 8
        self.o1_type_of_primary_tumor_num_classes = 8
        self.model_type_of_primary_tumor, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model type of primary tumor Finished")
        
        self.args.data.o1_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.o1_lung_SCLC_or_NSCLC_feature_dim = 4
        self.o1_lung_SCLC_or_NSCLC_num_classes = 4
        self.model_lung_SCLC_or_NSCLC, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model lung SCLC or NSCLC Finished")
        
        self.args.data.o1_extra_feature_dim = 6
        self.args.model.num_classes = 6
        self.o1_lung_EGFR_feature_dim = 6
        self.o1_lung_EGFR_num_classes = 6
        self.model_lung_EGFR, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model lung EGFR Finished")
        
        self.args.data.o1_extra_feature_dim = 6
        self.args.model.num_classes = 6
        self.o1_lung_ROS1_feature_dim = 6
        self.o1_lung_ROS1_num_classes = 6
        self.model_lung_ROS1, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model lung ROS1 Finished")
        
        self.args.data.o1_extra_feature_dim = 6
        self.args.model.num_classes = 6
        self.o1_lung_ALK_feature_dim = 6
        self.o1_lung_ALK_num_classes = 6
        self.model_lung_ALK, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model lung ALK Finished")
        
        self.args.data.o1_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.o1_breast_ER_feature_dim = 4
        self.o1_breast_ER_num_classes = 4
        self.model_breast_ER, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model breast ER Finished")
        
        self.args.data.o1_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.o1_breast_PR_feature_dim = 4
        self.o1_breast_PR_num_classes = 4
        self.model_breast_PR, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model breast PR Finished")
        
        self.args.data.o1_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.o1_breast_HER2_feature_dim = 4
        self.o1_breast_HER2_num_classes = 4
        self.model_breast_HER2, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model breast HER2 Finished")
        
        self.args.data.o1_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.o1_breast_AR_feature_dim = 4
        self.o1_breast_AR_num_classes = 4
        self.model_breast_AR, self.embed_dim = build_o1_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        self.logger.info("Model breast AR Finished")
        
        
        self.model_dict = {
            'bone_lesion': self.model_bone_lesion,
            'benign_malignant': self.model_benign_malignant,
            'primary_metastasis': self.model_primary_metastasis,
            'osteoblastic': self.model_osteoblastic,
            'osteolytic': self.model_osteolytic,
            'spinal_cord_compression': self.model_spinal_cord_compression,
            'thrombosis': self.model_thrombosis,
            'pathological_fracture': self.model_pathological_fracture,
            'renal_insufficiency': self.model_renal_insufficiency,
            'hypercalcemia': self.model_hypercalcemia,
            'type_of_primary_tumor': self.model_type_of_primary_tumor,
            'lung_SCLC_or_NSCLC': self.model_lung_SCLC_or_NSCLC,
            'lung_EGFR': self.model_lung_EGFR,
            'lung_ROS1': self.model_lung_ROS1,
            'lung_ALK': self.model_lung_ALK,
            'breast_ER': self.model_breast_ER,
            'breast_PR': self.model_breast_PR,
            'breast_HER2': self.model_breast_HER2,
        }
        
        self.feature_dim_dict = {
            'bone_lesion': self.o1_bone_lesion_feature_dim,
            'benign_malignant': self.o1_benign_malignant_feature_dim,
            'primary_metastasis': self.o1_primary_metastasis_feature_dim,
            'osteoblastic': self.o1_osteoblastic_feature_dim,
            'osteolytic': self.o1_osteolytic_feature_dim,
            'spinal_cord_compression': self.o1_spinal_cord_compression_feature_dim,
            'thrombosis': self.o1_thrombosis_feature_dim,
            'pathological_fracture': self.o1_pathological_fracture_feature_dim,
            'renal_insufficiency': self.o1_renal_insufficiency_feature_dim,
            'hypercalcemia': self.o1_hypercalcemia_feature_dim,
            'type_of_primary_tumor': self.o1_type_of_primary_tumor_feature_dim,
            'lung_SCLC_or_NSCLC': self.o1_lung_SCLC_or_NSCLC_feature_dim,
            'lung_EGFR': self.o1_lung_EGFR_feature_dim,
            'lung_ROS1': self.o1_lung_ROS1_feature_dim,
            'lung_ALK': self.o1_lung_ALK_feature_dim,
            'breast_ER': self.o1_breast_ER_feature_dim,
            'breast_PR': self.o1_breast_PR_feature_dim,
            'breast_HER2': self.o1_breast_HER2_feature_dim,
        }
        
        self.num_classes_dict = {
            'bone_lesion': self.o1_bone_lesion_num_classes,
            'benign_malignant': self.o1_benign_malignant_num_classes,
            'primary_metastasis': self.o1_primary_metastasis_num_classes,
            'osteoblastic': self.o1_osteoblastic_num_classes,
            'osteolytic': self.o1_osteolytic_num_classes,
            'spinal_cord_compression': self.o1_spinal_cord_compression_num_classes,
            'thrombosis': self.o1_thrombosis_num_classes,
            'pathological_fracture': self.o1_pathological_fracture_num_classes,
            'renal_insufficiency': self.o1_renal_insufficiency_num_classes,
            'hypercalcemia': self.o1_hypercalcemia_num_classes,
            'type_of_primary_tumor': self.o1_type_of_primary_tumor_num_classes,
            'lung_SCLC_or_NSCLC': self.o1_lung_SCLC_or_NSCLC_num_classes,
            'lung_EGFR': self.o1_lung_EGFR_num_classes,
            'lung_ROS1': self.o1_lung_ROS1_num_classes,
            'lung_ALK': self.o1_lung_ALK_num_classes,
            'breast_ER': self.o1_breast_ER_num_classes,
            'breast_PR': self.o1_breast_PR_num_classes,
        }
        
    def _load_single_model_checkpoint(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'])
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        return model
    
    def load_checkpoint(self):
        o1_fold_num = self.args.data.o1_fold_num
        for key_name in self.model_dict.keys():
            self.model_dict[key_name] = self._load_single_model_checkpoint(self.model_dict[key_name], o1_checkpoint_dict[key_name][o1_fold_num]) 
    
    def setup_model(self):
        for key_name in self.model_dict.keys():
            self.model_dict[key_name] = self.model_dict[key_name].cpu()
            
    @torch.no_grad()
    def val(self, n_epoch, split='val'):
        val_results_dict = {key_name: {} for key_name in self.model_dict.keys()}
        dataloader = self.dataloaders_dict[split]
        self.logger.info(f'Start evaluate {split} epoch {n_epoch}')
        
        if n_epoch == 0:
            pass
        else:
            total_pred_before = np.load(os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch-1}.npz'), allow_pickle=True)
            total_pred_before = total_pred_before['arr_0'].item()
            total_pred_feature = np.zeros([len(self.datasets_dict[split]), 26], dtype=np.float32) - 1
            for i, model_name in enumerate(self.model_dict.keys()):
                idx = 0
                data_dict = total_pred_before[model_name]
                for tiaoma in data_dict.keys():
                    for xulie in data_dict[tiaoma].keys():
                        for png_name in data_dict[tiaoma][xulie].keys():
                            if i <= 7 or i >=17:
                                total_pred_feature[idx, i] = data_dict[tiaoma][xulie][png_name]['pred']
                            elif i == 8:
                                total_pred_feature[idx, i:16] = data_dict[tiaoma][xulie][png_name]['pred']
                            idx += 1

        o1_use_feature_idx_dict = {
            'bone_lesion': [1,2,8,9,10,11,12,13,14,15,3,4,16,5,6,7],
            'benign_malignant': [0,2,8,9,10,11,12,13,14,15,3,4,5,6,7],
            'primary_metastasis': [0,1,8,9,10,11,12,13,14,15,3,4,5,6,7],
            'osteoblastic': [0,1,2,8,9,10,11,12,13,14,15,4,5,6,7],
            'osteolytic': [0,1,2,8,9,10,11,12,13,14,15,3,5,6,7],
            'spinal_cord_compression': [0,1,2,8,9,10,11,12,13,14,15,3,4,6,7],
            'thrombosis': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,7],
            'pathological_fracture': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,6],
            'renal_insufficiency': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,6],
            'hypercalcemia': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,6],
            'type_of_primary_tumor': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,6],
            'lung_SCLC_or_NSCLC': [3,4,5,6], # 4
            'lung_EGFR': [21,20,3,4,5,6], # 6
            'lung_ROS1': [19,21,3,4,5,6], # 6
            'lung_ALK': [19,20,3,4,5,6], # 6
            'breast_ER': [3,4,5,6], # 4
            'breast_PR': [3,4,5,6], # 4
            'breast_HER2': [3,4,5,6], # 4
            'breast_AR': [3,4,5,6], # 4
        }
        
        with tqdm(total=len(dataloader)*len(list(self.model_dict.keys())), desc=f"{split} Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            
            epoch_loss = 0.0
            for model_name in self.model_dict.keys():
                model = self.model_dict[model_name]
                model = model.cuda()
                model.eval()
                data_idx = 0
                for i, data_dict in enumerate(dataloader):
                    if len(data_dict) == 5:
                        raw_images, labels, _, image_name, study_series_names = data_dict['image'], data_dict['label'], data_dict['feature'], data_dict['image_name'], data_dict['study_series_name']
                    else:
                        raise ValueError(f"Invalid data_dict: {data_dict}")
                    raw_images = raw_images.to(self.device)
                    
                    if n_epoch == 0:
                        extra_features = torch.zeros([len(labels), self.feature_dim_dict[model_name]], dtype=torch.float32).to(self.device) - 1
                    else:
                        extra_features = torch.from_numpy(total_pred_feature[data_idx:data_idx+len(labels), o1_use_feature_idx_dict[model_name]]).to(self.device)
                    images = merge_o1_input_tensors(raw_images, extra_features)
                    
                    with self.amp_context:
                        outputs = model(images)
                        if self.num_classes_dict[model_name] == 1:
                            outputs = torch.sigmoid(outputs)
                        else:
                            outputs = torch.softmax(outputs, dim=1)
                        
                    if len(data_dict) == 5:
                        for j in range(len(study_series_names)):
                            study_id, series_id = study_series_names[j].split('_')
                            if study_id not in val_results_dict[model_name].keys():
                                val_results_dict[model_name][study_id] = {}
                            if series_id not in val_results_dict[model_name][study_id].keys():
                                val_results_dict[model_name][study_id][series_id] = {}                        
                            val_results_dict[model_name][study_id][series_id][image_name[j]] = {'pred': outputs[j].cpu().numpy()} 
                    
                    pbar.update(1)
                    data_idx += len(labels)
                model = model.cpu()
            save_file_name = os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch}.npz')
            np.savez(save_file_name, val_results_dict)
            self.logger.info(f"Save prediction results to {save_file_name}")
            self.logger.info(f'Finish evaluate {split} epoch {n_epoch}')
            return True

    def run(self):
        for n_epoch in range(self.start_epoch, self.total_epoch):
            self.val(n_epoch, split='val')