from .base_trainer import BaseTrainer
from ..models import build_bonecot_finetune_model_from_cfg
import torch
import traceback
from tqdm import tqdm
import os
import numpy as np

class BoneCoT_Finetune_Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def initlize(self):
        self.get_logger()
        self.get_criterion()
        
        self.get_data()
        
        self.get_model()
        self.load_checkpoint()
        self.get_metric()
        
        self.get_optimizer()
        self.get_scheduler()
        
        self.setup_model()
    
    def get_model(self):
        self.model, self.embed_dim = build_bonecot_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        
    def get_optimizer(self):
        params_group = self.model.get_optimizer_param_groups(self.args.optim.backbone_lr, self.args.optim.lr)
        if self.args.optim.optimizer.lower() in ['adam', 'adamw']:
            self.optimizer = torch.optim.AdamW(params_group, weight_decay=self.args.optim.weight_decay, betas=(self.args.optim.adamw_beta1, self.args.optim.adamw_beta2))
        elif self.args.optim.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params_group, weight_decay=self.args.optim.weight_decay, momentum=self.args.optim.sgd_momentum)
        else:
            raise ValueError(f"Optimizer {self.args.optim.optimizer} not supported.")    
    
    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.args.model.pretrained_weights}")
        if self.args.model.pretrained_weights == "":
            self.logger.info("No checkpoint loaded.")
            return
        try:
            model_state_dict = self.model.backbone.feature_model.state_dict()
            checkpoint = torch.load(self.args.model.pretrained_weights, map_location='cpu', weights_only=True)
            if 'pos_embed' in checkpoint.keys():
                if checkpoint['pos_embed'].shape != model_state_dict['pos_embed'].shape:
                    checkpoint['pos_embed'] = model_state_dict['pos_embed']
                self.model.backbone.feature_model.load_state_dict(checkpoint, strict=True)
            elif 'teacher' in checkpoint.keys():
                for key, values in checkpoint['teacher'].items():
                    if 'backbone' in key:
                        model_state_dict[key.replace('backbone.', '')] = values
                self.model.backbone.feature_model.load_state_dict(model_state_dict, strict=True)
            else:
                raise ValueError("Invalid checkpoint")
            self.logger.info(f"Loaded pretrained model from {self.args.model.pretrained_weights}")
        except Exception as e:
            self.logger.error(f"Failed to load pretrained model from {self.args.model.pretrained_weights}")
            self.logger.error(f"Exception: {e}")
            self.logger.error(traceback.format_exc())
            
        ckpt_list = self.args.model.extra_model_ckpt_list
        model_dict_list = []
        for extra_model_ckpt in ckpt_list:
            checkpoint = torch.load(extra_model_ckpt, map_location='cpu', weights_only=False)
            print(checkpoint['model'].keys())
            if 'model' in checkpoint:
                checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
                model_dict_list.append(checkpoint['model'])
                self.logger.info(f"Loaded extra model from {extra_model_ckpt}")
        if model_dict_list:
            self.model.init_extra_token_linear_layers(model_dict_list)
            self.logger.info(f"Loaded extra model from {ckpt_list}")
            
    def train(self, n_epoch):
        self.model.train()
        self.logger.info(f'Start training at epoch {n_epoch}/{self.total_epoch}')
        with tqdm(total=len(self.train_dataloader), desc=f"Train Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            epoch_loss = 0.0
            for i, data_dict in enumerate(self.train_dataloader):
                if len(data_dict) == 2:
                    images, labels = data_dict['image'], data_dict['label']
                elif len(data_dict) == 3:
                    images, labels, study_series_names = data_dict['image'], data_dict['label'], data_dict['study_series_name']
                elif len(data_dict) == 4:
                    images, labels, image_name, study_series_names = data_dict['image'], data_dict['label'], data_dict['image_name'], data_dict['study_series_name']
                else:
                    raise ValueError(f"Invalid data_dict: {data_dict}")

                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                with self.amp_context:
                    outputs = self.model(images)
                    if self.args.optim.loss_type in ['BCE', 'BCE_Focal']:
                        labels = labels.float()
                        if outputs.shape != labels.shape and self.num_classes == 1:
                            labels = labels.unsqueeze(1)
                    else:
                        labels = labels.long()
                    loss = self.criterion(outputs, labels)

                self.metric.update(outputs.detach().cpu(), labels)
                epoch_loss += loss.item()

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            
                self.log_tensorboard.add_scalar('train_loss', loss.item(), n_epoch * len(self.train_dataloader) + i)

                if self.max_step_per_epoch != -1 and i >= self.max_step_per_epoch:
                    break
                
            avg_loss = epoch_loss / len(self.train_dataloader)
            metrics_result = self.metric.results()
            for key, value in metrics_result.items():
                self.log_tensorboard.add_scalar(f'train_{key}', value, n_epoch)
            
            self.logger.info(f'Finish training epoch {n_epoch}')
            return avg_loss, metrics_result
        
    @torch.no_grad()
    def val(self, n_epoch, split='val'):
        val_results_dict = {}
        self.model.eval()
        dataloader = self.dataloaders_dict[split]
        self.logger.info(f'Start evaluate {split} epoch {n_epoch}')
        with tqdm(total=len(dataloader), desc=f"{split} Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            epoch_loss = 0.0
            for i, data_dict in enumerate(dataloader):
                if len(data_dict) == 2:
                    images, labels = data_dict['image'], data_dict['label']
                elif len(data_dict) == 3:
                    images, labels, study_series_names = data_dict['image'], data_dict['label'], data_dict['study_series_name']
                elif len(data_dict) == 4:
                    images, labels, image_name, study_series_names = data_dict['image'], data_dict['label'], data_dict['image_name'], data_dict['study_series_name']
                else:
                    raise ValueError(f"Invalid data_dict: {data_dict}")
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with self.amp_context:
                    outputs = self.model(images)
                    if self.args.optim.loss_type in ['BCE', 'BCE_Focal']:
                        labels = labels.float()
                        if outputs.shape != labels.shape and self.num_classes == 1:
                            labels = labels.unsqueeze(1)
                    else:
                        labels = labels.long()
                    loss = self.criterion(outputs, labels)
                    
                for j in range(len(study_series_names)):
                    study_id, series_id = study_series_names[j].split('_')
                    if study_id not in val_results_dict.keys():
                        val_results_dict[study_id] = {}
                    if series_id not in val_results_dict[study_id].keys():
                        val_results_dict[study_id][series_id] = {}                        
                    val_results_dict[study_id][series_id][image_name[j]] = {'label': labels[j].cpu().numpy(), 'pred': outputs[j].cpu().numpy()} 
                    
                self.metric.update(outputs.detach().cpu(), labels)
                epoch_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            
            avg_loss = epoch_loss / len(dataloader)
            metrics_result = self.metric.results()
            for key, value in metrics_result.items():
                self.log_tensorboard.add_scalar(f'{split}_{key}', value, n_epoch)
                
            save_file_name = os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch}.npz')
            np.savez(save_file_name, val_results_dict)
            self.logger.info(f"Save prediction results to {save_file_name}")
            self.logger.info(f'Finish evaluate {split} epoch {n_epoch}')
            return avg_loss, metrics_result
        
    def save_checkpoint(self, n_epoch, save_model=True):
        output_dir = os.path.join(self.args.output_dir, 'checkpoints')
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)
        if save_model:
            model_state_dict = {}
            for name, param in self.model.state_dict().items():
                if 'linear_classifier' in name:
                    model_state_dict[name] = param
                    
            save_dict = {
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': n_epoch,
                'results': self.results_dict
            }
        else:
            save_dict = {
                'epoch': n_epoch,
                'results': self.results_dict
            }
        
        torch.save(save_dict, os.path.join(output_dir, f'epoch_{n_epoch}.pth'))
        self.logger.info(f"Save checkpoint at epoch {n_epoch}")