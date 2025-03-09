from .base_trainer import BaseTrainer
from ..models import build_bonecot_linear_finetune_model_from_cfg
import torch
from torchvision import transforms
from ..data import BoneCoT_Inference_Dataset
import os
import numpy as np

def merge_bonecot_input_tensors(image, feature):
    _, _, height, width = image.shape
    expanded_feature = feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
    combined_tensor = torch.cat([image, expanded_feature], dim=1)
    return combined_tensor

class BoneCoT_Eval_Trainer(BaseTrainer):
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
        self.test_transform = transforms.Compose([
            transforms.Resize(self.args.crops.global_crops_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_transform = self.test_transform
        self.test_dataset = BoneCoT_Inference_Dataset(self.args.data.test_dataset, transform=self.test_transform)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.val_dataset = self.test_dataset
        self.val_dataloader = self.test_dataloader
        self.train_dataset = self.val_dataset
        self.train_dataloader = self.val_dataloader
        
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
        
        self.logger.info(f"Test dataset: {self.args.data.test_dataset} / Length: {len(self.test_dataset)} / loader: {len(self.test_dataloader)}")
    
    def get_model(self):
        self.use_cot_relative_model = {}
        for task_name in self.args.model.cot_relative_model_dict.keys():
            if os.path.exists(self.args.model.cot_relative_model_dict[task_name]):
                self.use_cot_relative_model[task_name] = self.args.model.cot_relative_model_dict[task_name]
        self.logger.info(f"Use CoT relative model for inference: {self.use_cot_relative_model.keys()}")
        self.args.data.bonecot_extra_feature_dim = 16
        self.args.model.num_classes = 1
        self.bonecot_bone_lesion_feature_dim = 16
        self.bonecot_bone_lesion_num_classes = 1
        
        if 'bone_lesion' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['bone_lesion']):
            self.model_bone_lesion, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model bone lesion Finished")
        else:
            self.model_bone_lesion = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_benign_malignant_feature_dim = 15
        self.bonecot_benign_malignant_num_classes = 1
        if 'benign_malignant' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['benign_malignant']):
            self.model_benign_malignant, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model benign malignant Finished")
        else:
            self.model_benign_malignant = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_primary_metastatic_feature_dim = 15
        self.bonecot_primary_metastatic_num_classes = 1
        if 'primary_metastatic' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['primary_metastatic']):
            self.model_primary_metastatic, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model primary metastatic Finished")
        else:
            self.model_primary_metastatic = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_osteoblastic_feature_dim = 15
        self.bonecot_osteoblastic_num_classes = 1
        if 'osteoblastic' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['osteoblastic']):
            self.model_osteoblastic, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model osteoblastic Finished")
        else:
            self.model_osteoblastic = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_osteolytic_feature_dim = 15
        self.bonecot_osteolytic_num_classes = 1
        if 'osteolytic' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['osteolytic']):
            self.model_osteolytic, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model osteolytic Finished")
        else:
            self.model_osteolytic = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_spinal_cord_compression_feature_dim = 15
        self.bonecot_spinal_cord_compression_num_classes = 1
        if 'spinal_cord_compression' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['spinal_cord_compression']):
            self.model_spinal_cord_compression, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model spinal cord compression Finished")
        else:
            self.model_spinal_cord_compression = None
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_thrombosis_feature_dim = 15
        self.bonecot_thrombosis_num_classes = 1
        if 'thrombosis' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['thrombosis']):
            self.model_thrombosis, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model thrombosis Finished")
        else:
            self.model_thrombosis = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_pathological_fracture_feature_dim = 15
        self.bonecot_pathological_fracture_num_classes = 1
        if 'pathological_fracture' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['pathological_fracture']):
            self.model_pathological_fracture, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model pathological fracture Finished")
        else:
            self.model_pathological_fracture = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_renal_insufficiency_feature_dim = 15
        self.bonecot_renal_insufficiency_num_classes = 1
        if 'renal_insufficiency' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['renal_insufficiency']):
            self.model_renal_insufficiency, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model renal insufficiency Finished")
        else:
            self.model_renal_insufficiency = None
        
        
        self.args.data.bonecot_extra_feature_dim = 15
        self.args.model.num_classes = 1
        self.bonecot_hypercalcemia_feature_dim = 15
        self.bonecot_hypercalcemia_num_classes = 1
        if 'hypercalcemia' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['hypercalcemia']):
            self.model_hypercalcemia, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model hypercalcemia Finished")
        else:
            self.model_hypercalcemia = None
        
        
        self.args.data.bonecot_extra_feature_dim = 8
        self.args.model.num_classes = 8
        self.bonecot_type_of_primary_tumor_feature_dim = 8
        self.bonecot_type_of_primary_tumor_num_classes = 8
        if 'type_of_primary_tumor' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['type_of_primary_tumor']):
            self.model_type_of_primary_tumor, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model type of primary tumor Finished")
        else:
            self.model_type_of_primary_tumor = None
        
        
        self.args.data.bonecot_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.bonecot_lung_SCLC_or_NSCLC_feature_dim = 4
        self.bonecot_lung_SCLC_or_NSCLC_num_classes = 4
        if 'lung_SCLC_or_NSCLC' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['lung_SCLC_or_NSCLC']):
            self.model_lung_SCLC_or_NSCLC, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model lung SCLC or NSCLC Finished")
        else:
            self.model_lung_SCLC_or_NSCLC = None
        
        
        self.args.data.bonecot_extra_feature_dim = 6
        self.args.model.num_classes = 6
        self.bonecot_lung_EGFR_feature_dim = 6
        self.bonecot_lung_EGFR_num_classes = 6
        if 'lung_EGFR' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['lung_EGFR']):
            self.model_lung_EGFR, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model lung EGFR Finished")
        else:
            self.model_lung_EGFR = None
        
        
        self.args.data.bonecot_extra_feature_dim = 6
        self.args.model.num_classes = 6
        self.bonecot_lung_ROS1_feature_dim = 6
        self.bonecot_lung_ROS1_num_classes = 6
        if 'lung_ROS1' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['lung_ROS1']):
            self.model_lung_ROS1, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model lung ROS1 Finished")
        else:
            self.model_lung_ROS1 = None
        
        
        self.args.data.bonecot_extra_feature_dim = 6
        self.args.model.num_classes = 6
        self.bonecot_lung_ALK_feature_dim = 6
        self.bonecot_lung_ALK_num_classes = 6
        if 'lung_ALK' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['lung_ALK']):
            self.model_lung_ALK, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model lung ALK Finished")
        else:
            self.model_lung_ALK = None
        
        
        self.args.data.bonecot_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.bonecot_breast_ER_feature_dim = 4
        self.bonecot_breast_ER_num_classes = 4
        if 'breast_ER' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['breast_ER']):
            self.model_breast_ER, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model breast ER Finished")
        else:
            self.model_breast_ER = None
        
        
        self.args.data.bonecot_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.bonecot_breast_PR_feature_dim = 4
        self.bonecot_breast_PR_num_classes = 4
        if 'breast_PR' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['breast_PR']):
            self.model_breast_PR, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model breast PR Finished")
        else:
            self.model_breast_PR = None
        
        
        self.args.data.bonecot_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.bonecot_breast_HER2_feature_dim = 4
        self.bonecot_breast_HER2_num_classes = 4
        if 'breast_HER2' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['breast_HER2']):
            self.model_breast_HER2, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model breast HER2 Finished")
        else:
            self.model_breast_HER2 = None
        
        
        self.args.data.bonecot_extra_feature_dim = 4
        self.args.model.num_classes = 4
        self.bonecot_breast_AR_feature_dim = 4
        self.bonecot_breast_AR_num_classes = 4
        if 'breast_AR' in self.use_cot_relative_model and os.path.exists(self.use_cot_relative_model['breast_AR']):
            self.model_breast_AR, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
            self.logger.info("Model breast AR Finished")
        else:
            self.model_breast_AR = None
        
        
        
        self.model_dict = {
            'bone_lesion': self.model_bone_lesion,
            'benign_malignant': self.model_benign_malignant,
            'primary_metastatic': self.model_primary_metastatic,
            'osteoblastic': self.model_osteoblastic,
            'osteolytic': self.model_osteolytic,
            'pathological_fracture': self.model_pathological_fracture,
            'spinal_cord_compression': self.model_spinal_cord_compression,
            'thrombosis': self.model_thrombosis,
            'type_of_primary_tumor': self.model_type_of_primary_tumor,
            'hypercalcemia': self.model_hypercalcemia,
            'lung_SCLC_or_NSCLC': self.model_lung_SCLC_or_NSCLC,
            'lung_EGFR': self.model_lung_EGFR,
            'lung_ROS1': self.model_lung_ROS1,
            'lung_ALK': self.model_lung_ALK,
            'breast_ER': self.model_breast_ER,
            'breast_PR': self.model_breast_PR,
            'breast_HER2': self.model_breast_HER2,
            'breast_AR': self.model_breast_AR,
            'renal_insufficiency': self.model_renal_insufficiency,
        }
        
        self.feature_dim_dict = {
            'bone_lesion': self.bonecot_bone_lesion_feature_dim,
            'benign_malignant': self.bonecot_benign_malignant_feature_dim,
            'primary_metastatic': self.bonecot_primary_metastatic_feature_dim,
            'osteoblastic': self.bonecot_osteoblastic_feature_dim,
            'osteolytic': self.bonecot_osteolytic_feature_dim,
            'pathological_fracture': self.bonecot_pathological_fracture_feature_dim,
            'spinal_cord_compression': self.bonecot_spinal_cord_compression_feature_dim,
            'thrombosis': self.bonecot_thrombosis_feature_dim,
            'type_of_primary_tumor': self.bonecot_type_of_primary_tumor_feature_dim,
            'hypercalcemia': self.bonecot_hypercalcemia_feature_dim,
            'lung_SCLC_or_NSCLC': self.bonecot_lung_SCLC_or_NSCLC_feature_dim,
            'lung_EGFR': self.bonecot_lung_EGFR_feature_dim,
            'lung_ROS1': self.bonecot_lung_ROS1_feature_dim,
            'lung_ALK': self.bonecot_lung_ALK_feature_dim,
            'breast_ER': self.bonecot_breast_ER_feature_dim,
            'breast_PR': self.bonecot_breast_PR_feature_dim,
            'breast_HER2': self.bonecot_breast_HER2_feature_dim,
            'breast_AR': self.bonecot_breast_AR_feature_dim,
            'renal_insufficiency': self.bonecot_renal_insufficiency_feature_dim,
        }
        
        self.num_classes_dict = {
            'bone_lesion': self.bonecot_bone_lesion_num_classes,
            'benign_malignant': self.bonecot_benign_malignant_num_classes,
            'primary_metastatic': self.bonecot_primary_metastatic_num_classes,
            'osteoblastic': self.bonecot_osteoblastic_num_classes,
            'osteolytic': self.bonecot_osteolytic_num_classes,
            'pathological_fracture': self.bonecot_pathological_fracture_num_classes,
            'spinal_cord_compression': self.bonecot_spinal_cord_compression_num_classes,
            'thrombosis': self.bonecot_thrombosis_num_classes,
            'type_of_primary_tumor': self.bonecot_type_of_primary_tumor_num_classes,
            'hypercalcemia': self.bonecot_hypercalcemia_num_classes,
            'lung_SCLC_or_NSCLC': self.bonecot_lung_SCLC_or_NSCLC_num_classes,
            'lung_EGFR': self.bonecot_lung_EGFR_num_classes,
            'lung_ROS1': self.bonecot_lung_ROS1_num_classes,
            'lung_ALK': self.bonecot_lung_ALK_num_classes,
            'breast_ER': self.bonecot_breast_ER_num_classes,
            'breast_PR': self.bonecot_breast_PR_num_classes,
            'breast_HER2': self.bonecot_breast_HER2_num_classes,
            'breast_AR': self.bonecot_breast_AR_num_classes,
            'renal_insufficiency': self.bonecot_renal_insufficiency_num_classes,
        }
        
    def _load_single_model_checkpoint(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'])
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        return model
    
    # can change the checkpoint path here
    def load_checkpoint(self):
        for key_name in self.model_dict.keys():
            if key_name not in self.use_cot_relative_model.keys():
                continue
            if os.path.exists(self.use_cot_relative_model[key_name]):
                self.model_dict[key_name] = self._load_single_model_checkpoint(self.model_dict[key_name], self.use_cot_relative_model[key_name]) 

    
    def setup_model(self):
        for key_name in self.model_dict.keys():
            if self.model_dict[key_name] is not None:
                self.model_dict[key_name] = self.model_dict[key_name].cpu()
    
    @torch.no_grad()
    def val(self, n_epoch, split='val'):
        val_results_dict = {key_name: {} for key_name in self.model_dict.keys()}
        dataloader = self.dataloaders_dict[split]
        self.logger.info(f'Start BoneCoT inference on {split} dataset with round {n_epoch+1}')
        
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
                                pred = data_dict[tiaoma][xulie][png_name]['pred']
                                # Handle both scalar and array cases
                                if isinstance(pred, np.ndarray):
                                    # Check array size before calling item()
                                    if pred.size == 1:
                                        total_pred_feature[idx, i] = float(pred.item())
                                    else:
                                        # Take first element if array has multiple values
                                        total_pred_feature[idx, i] = float(pred.flatten()[0])
                                else:
                                    total_pred_feature[idx, i] = float(pred)
                            elif i == 8:
                                pred_array = data_dict[tiaoma][xulie][png_name]['pred']
                                total_pred_feature[idx, i:16] = pred_array.flatten()
                            idx += 1

        bonecot_use_feature_idx_dict = {
            'bone_lesion': [1,2,8,9,10,11,12,13,14,15,3,4,16,5,6,7],
            'benign_malignant': [0,2,8,9,10,11,12,13,14,15,3,4,5,6,7],
            'primary_metastatic': [0,1,8,9,10,11,12,13,14,15,3,4,5,6,7],
            'osteoblastic': [0,1,2,8,9,10,11,12,13,14,15,4,5,6,7],
            'osteolytic': [0,1,2,8,9,10,11,12,13,14,15,3,5,6,7],
            'pathological_fracture': [0,1,2,8,9,10,11,12,13,14,15,3,4,6,7],
            'spinal_cord_compression': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,7],
            'thrombosis': [0,1,8,9,10,11,12,13,14,15,2,3,4,5,6],
            'type_of_primary_tumor': [0,1,2,3,4,5,6,7],
            'hypercalcemia': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,6],
            'lung_SCLC_or_NSCLC': [3,4,5,6],
            'lung_EGFR': [21,20,3,4,5,6],
            'lung_ROS1': [19,21,3,4,5,6],
            'lung_ALK': [19,20,3,4,5,6],
            'breast_ER': [3,4,5,6],
            'breast_PR': [3,4,5,6],
            'breast_HER2': [3,4,5,6],
            'breast_AR': [3,4,5,6],
            'renal_insufficiency': [0,1,2,8,9,10,11,12,13,14,15,3,4,5,6],
        }
        
        for model_name in self.model_dict.keys():
            model = self.model_dict[model_name]
            data_idx = 0
            if model is None:
                # If model is None, still iterate through dataloader but use default -1 predictions
                for i, data_dict in enumerate(dataloader):
                    if len(data_dict) == 5:
                        labels, image_name, study_series_names = data_dict['label'], data_dict['image_name'], data_dict['study_series_name']
                    else:
                        raise ValueError(f"Invalid data_dict: {data_dict}")
                        
                    # Use -1 as default prediction
                    outputs = torch.ones(len(labels), 1) * -1 if self.num_classes_dict[model_name] == 1 else torch.ones(len(labels), self.num_classes_dict[model_name]) * -1
                    
                    for j in range(len(study_series_names)):
                        study_id, series_id = study_series_names[j].split('_')
                        if study_id not in val_results_dict[model_name].keys():
                            val_results_dict[model_name][study_id] = {}
                        if series_id not in val_results_dict[model_name][study_id].keys():
                            val_results_dict[model_name][study_id][series_id] = {}                        
                        val_results_dict[model_name][study_id][series_id][image_name[j]] = {'pred': outputs[j].numpy()}
                    
                    data_idx += len(labels)
            else:
                model = model.cuda()
                model.eval()
                for i, data_dict in enumerate(dataloader):
                    if len(data_dict) == 5:
                        raw_images, labels, _, image_name, study_series_names = data_dict['image'], data_dict['label'], data_dict['feature'], data_dict['image_name'], data_dict['study_series_name']
                    else:
                        raise ValueError(f"Invalid data_dict: {data_dict}")
                    raw_images = raw_images.to(self.device)
                    
                    if n_epoch == 0:
                        extra_features = torch.zeros([len(labels), self.feature_dim_dict[model_name]], dtype=torch.float32).to(self.device) - 1
                    else:
                        extra_features = torch.from_numpy(total_pred_feature[data_idx:data_idx+len(labels), bonecot_use_feature_idx_dict[model_name]]).to(self.device)
                    images = merge_bonecot_input_tensors(raw_images, extra_features)
                    
                    with self.amp_context:
                        outputs = model(images)
                        if self.num_classes_dict[model_name] == 1:
                            preds = torch.sigmoid(outputs)
                        else:
                            preds = torch.softmax(outputs, dim=1)
                        
                    if len(data_dict) == 5:
                        for j in range(len(study_series_names)):
                            study_id, series_id = study_series_names[j].split('_')
                            if study_id not in val_results_dict[model_name].keys():
                                val_results_dict[model_name][study_id] = {}
                            if series_id not in val_results_dict[model_name][study_id].keys():
                                val_results_dict[model_name][study_id][series_id] = {}                        
                            val_results_dict[model_name][study_id][series_id][image_name[j]] = {'pred': preds[j].cpu().numpy(), 'outputs': outputs[j].cpu().numpy()} 
                    
                    data_idx += len(labels)
                model = model.cpu()
        save_file_name = os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch}.npz')
        np.savez(save_file_name, val_results_dict)
        return True

    def run(self):
        for n_epoch in range(self.start_epoch, self.total_epoch):
            self.val(n_epoch, split='test')
        self.logger.info(f"Inference finished")