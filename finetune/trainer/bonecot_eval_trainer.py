from .base_trainer import BaseTrainer
from ..models import build_bonecot_multi_round_inference_model_from_cfg, build_bonecot_relative_model
import torch
from torchvision import transforms
from ..data import BoneCoT_Inference_Dataset
import os
import numpy as np


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
        self.backbone_model, self.model, self.embed_dim = build_bonecot_multi_round_inference_model_from_cfg(self.args, only_teacher=True)
        self.relative_model_num = len(self.args.model.relative_task)
        self.relative_ckpt_dict = {}
        self.relative_model_dict = {}
        self.relative_model_config = {}
        # 读取对应每个relative_task的ckpt的
        for task_name in self.args.model.relative_task:
            if os.path.exists(self.args.model.model_ckpt_dict[task_name]):
                self.relative_ckpt_dict[task_name] = torch.load(self.args.model.model_ckpt_dict[task_name], map_location='cpu', weights_only=False)['model']
                print(f"Use relative model for inference: {task_name}")
                self.relative_model_config[task_name] = {
                    'extra_token_num': (self.relative_ckpt_dict[task_name]['linear_classifier_1.weight'].shape[1]//1536) - 5,
                    'use_n_blocks': 4,
                    'embed_dim': self.embed_dim,
                    'use_avgpool': True,
                    'num_classes': 1
                }
                if task_name == 'type_of_primary_tumor':
                    self.relative_model_config[task_name]['num_classes'] = 9
                self.relative_model_dict[task_name] = build_bonecot_relative_model(self.backbone_model, **self.relative_model_config[task_name])
        self.logger.info(f"Use relative model for inference: {self.relative_model_dict.keys()}")
    
    # can change the checkpoint path here
    def load_checkpoint(self):
        try:
            model_state_dict = self.model.backbone.feature_model.state_dict()
            checkpoint = torch.load(self.args.model.backbone_ckpt_path, map_location='cpu', weights_only=True)
            if 'pos_embed' in checkpoint.keys():
                if checkpoint['pos_embed'].shape != model_state_dict['pos_embed'].shape:
                    checkpoint['pos_embed'] = model_state_dict['pos_embed']
                self.model.backbone.feature_model.load_state_dict(checkpoint, strict=True)
            elif 'teacher' in checkpoint.keys():
                for key, values in checkpoint['teacher'].items():
                    if 'backbone' in key:
                        model_state_dict[key.replace('backbone.', '')] = values
                self.model.backbone.feature_model.load_state_dict(model_state_dict, strict=True)  # backbone部分的都一起导入
            else:
                raise ValueError("Invalid checkpoint")
        except Exception as e:
            self.logger.error(f"Failed to load pretrained model from {self.args.model.backbone_ckpt_path}")
            self.logger.error(f"Exception: {e}")
            self.logger.error(traceback.format_exc())
        self.logger.info(f"Load backbone checkpoint from {self.args.model.backbone_ckpt_path}")
        self.main_task_ckpt = torch.load(self.args.model.model_ckpt_dict[self.args.model.main_task], map_location='cpu', weights_only=False)['model']
        self.load_classifier_ckpt(self.model, self.main_task_ckpt)
        self.logger.info(f"Load main task checkpoint from {self.args.model.model_ckpt_dict[self.args.model.main_task]}")
        for task_name in self.args.model.relative_task:
            self.load_classifier_ckpt(self.relative_model_dict[task_name], self.relative_ckpt_dict[task_name])
            self.logger.info(f"Load relative model checkpoint from {self.args.model.model_ckpt_dict[task_name]} for {task_name}")
            
    def load_classifier_ckpt(self, model, ckpt_dict):
        model_dict = model.state_dict()
        for key_name in ckpt_dict.keys():
            if key_name in model_dict.keys():
                model_dict[key_name] = ckpt_dict[key_name]
        model.load_state_dict(model_dict)
    
    @torch.no_grad()
    def val(self, n_epoch, split='val'):
        bonecot_use_feature_dict = {
            'primary_metastatic': ["osteoblastic", "osteolytic", "type_of_primary_tumor", "pathological_fracture", "spinal_cord_compression"],
            'osteoblastic': ["type_of_primary_tumor", "pathological_fracture"],
            'osteolytic': ["type_of_primary_tumor", "pathological_fracture"],
            'pathological_fracture': ["osteoblastic", "osteolytic", "type_of_primary_tumor"],
            'spinal_cord_compression': ["osteoblastic", "osteolytic", "type_of_primary_tumor", "pathological_fracture"],
            'type_of_primary_tumor': ["osteoblastic", "osteolytic"],
        }
        hidden_feature_task_name_idx = {
            'primary_metastatic': 0,
            'osteoblastic': 1,
            'osteolytic': 2,
            'pathological_fracture': 3,
            'spinal_cord_compression': 4,
            'type_of_primary_tumor': 5,
        }
        val_results_dict = {key_name: {} for key_name in self.relative_model_dict.keys()}
        val_results_dict[self.args.model.main_task] = {}
        dataloader = self.dataloaders_dict[split]
        self.logger.info(f'Start BoneCoT inference on {split} dataset with round {n_epoch+1}')
        
        if n_epoch == 0:
            total_hidden_feature_before = np.zeros([len(self.datasets_dict[split]), len(self.relative_model_dict.keys()) + 1, self.embed_dim], dtype=np.float32)
        else:
            total_results_before = np.load(os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch-1}.npz'), allow_pickle=True)
            total_results_before = total_results_before['arr_0'].item()
            total_hidden_feature_before = np.zeros([len(self.datasets_dict[split]), len(self.relative_model_dict.keys()) + 1, self.embed_dim], dtype=np.float32)
            for i, task_name in enumerate([self.args.model.main_task] + self.args.model.relative_task):
                data_idx = 0
                for tiaoma in total_results_before[task_name].keys():
                    for xulie in total_results_before[task_name][tiaoma].keys():
                        for png_name in total_results_before[task_name][tiaoma][xulie].keys():
                            total_hidden_feature_before[data_idx, i] = total_results_before[task_name][tiaoma][xulie][png_name]['hidden_features']
                            data_idx += 1
                
        total_hidden_feature = np.zeros([len(self.datasets_dict[split]), len(self.relative_model_dict.keys()) + 1, self.embed_dim], dtype=np.float32)
        # First inference for the main task
        for i, data_dict in enumerate(dataloader):
            self.model.cuda()
            self.model.eval()
            if len(data_dict) == 5:
                raw_images, labels, _, image_name, study_series_names = data_dict['image'], data_dict['label'], data_dict['feature'], data_dict['image_name'], data_dict['study_series_name']
            else:
                raise ValueError(f"Invalid data_dict: {data_dict}")
            raw_images = raw_images.to(self.device)
            
            # Get relative hidden features
            hidden_feature_list = []
            for relative_task_name in bonecot_use_feature_dict[self.args.model.main_task]:
                hidden_feature_list.append(total_hidden_feature_before[i, hidden_feature_task_name_idx[relative_task_name]])
            hidden_feature_concate = np.concatenate(hidden_feature_list, axis=0)
            hidden_feature_concate = torch.from_numpy(hidden_feature_concate).to(self.device)
            hidden_feature_concate = hidden_feature_concate.unsqueeze(0)
            with self.amp_context:
                outputs, hidden_features = self.model(raw_images, extra_hidden_features=hidden_feature_concate, hidden_output=True)
                
                if self.args.model.main_task != 'type_of_primary_tumor':
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                total_hidden_feature[i, 0] = hidden_features.detach().cpu().numpy()
                if len(data_dict) == 5:
                    for j in range(len(study_series_names)):
                        study_id, series_id = study_series_names[j].split('_')
                        if study_id not in val_results_dict[self.args.model.main_task].keys():
                            val_results_dict[self.args.model.main_task][study_id] = {}
                        if series_id not in val_results_dict[self.args.model.main_task][study_id].keys():
                            val_results_dict[self.args.model.main_task][study_id][series_id] = {}                        
                        val_results_dict[self.args.model.main_task][study_id][series_id][image_name[j]] = {'pred': preds[j].cpu().numpy(), 'outputs': outputs[j].cpu().numpy(), 'hidden_features': hidden_features[j].cpu().numpy()} 
                
            for j, model_name in enumerate(self.relative_model_dict.keys()):
                self.relative_model_dict[model_name].cuda()
                self.relative_model_dict[model_name].eval()
                # Get relative hidden features
                hidden_feature_list = []
                for relative_task_name in bonecot_use_feature_dict[model_name]:
                    hidden_feature_list.append(total_hidden_feature_before[i, hidden_feature_task_name_idx[relative_task_name]])
                hidden_feature_concate = np.concatenate(hidden_feature_list, axis=0)
                hidden_feature_concate = torch.from_numpy(hidden_feature_concate).to(self.device)
                hidden_feature_concate = hidden_feature_concate.unsqueeze(0)
                with self.amp_context:
                    outputs, hidden_features = self.relative_model_dict[model_name](raw_images, extra_hidden_features=hidden_feature_concate, hidden_output=True)
                    
                    if model_name != 'type_of_primary_tumor':
                        preds = torch.sigmoid(outputs)
                    else:
                        preds = torch.softmax(outputs, dim=1)
                    
                    total_hidden_feature[i, j+1] = hidden_features.detach().cpu().numpy()        
                    
                    if len(data_dict) == 5:
                        for j in range(len(study_series_names)):
                            study_id, series_id = study_series_names[j].split('_')
                            if study_id not in val_results_dict[model_name].keys():
                                val_results_dict[model_name][study_id] = {}
                            if series_id not in val_results_dict[model_name][study_id].keys():
                                val_results_dict[model_name][study_id][series_id] = {}                        
                            val_results_dict[model_name][study_id][series_id][image_name[j]] = {'pred': preds[j].cpu().numpy(), 'outputs': outputs[j].cpu().numpy(), 'hidden_features': hidden_features[j].cpu().numpy()} 

        save_file_name = os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch}.npz')
        np.savez(save_file_name, val_results_dict)
        return True

    def run(self):
        for n_epoch in range(self.start_epoch, self.total_epoch):
            self.val(n_epoch, split='test')
        self.logger.info(f"Inference finished")