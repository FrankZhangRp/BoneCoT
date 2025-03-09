from .base_trainer import BaseTrainer
from ..models import build_bonecot_linear_finetune_model_from_cfg, build_bonecot_single_linear_finetune_model_from_cfg
import torch
import traceback
from torchvision import transforms
from ..data import BoneCoT_Inference_Dataset
from tqdm import tqdm
import os
import numpy as np

def merge_bonecot_input_tensors(image, feature):
    """
    Merge image and feature tensors by expanding feature to [batch_size, feature_channels, height, width] and concatenating with image.
    image: [batch_size, 3, height, width]
    feature: [batch_size, feature_channels]
    Returns merged tensor: [batch_size, 3 + feature_channels, height, width]
    """
    # Get image height and width
    _, _, height, width = image.shape
    
    # Expand feature tensor to match image spatial dimensions
    expanded_feature = feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
    
    # Concatenate image and expanded_feature along channel dimension
    combined_tensor = torch.cat([image, expanded_feature], dim=1)
    
    return combined_tensor

class BoneCoT_Finetune_Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def initlize(self):
        '''
        Initialize in the new order
        '''
        # Function flow
        self.get_logger()
        self.get_criterion()
        
        # Data
        self.get_data()
        
        # Model
        self.get_model()
        # Import historical information - could be backbone parameters, entire model parameters, or checkpoint for resuming
        self.load_checkpoint()
        self.get_metric()
        # Optimization
        self.get_optimizer()
        self.get_scheduler()
        
        # Finally wrap model with DP
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
        self.train_dataset = BoneCoT_Inference_Dataset(self.args.data.train_dataset, transform=self.train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        assert self.args.data.val_dataset is not None
        self.val_dataset = BoneCoT_Inference_Dataset(self.args.data.val_dataset, transform=self.test_transform)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        if self.args.data.test_dataset is not None:
            self.test_dataset = BoneCoT_Inference_Dataset(self.args.data.test_dataset, transform=self.test_transform)
        else:
            self.test_dataset = self.val_dataset
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        
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
        # Full parameter fine-tuning
        self.model, self.embed_dim = build_bonecot_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)
        
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
            # Several possibilities - first is ImageNet pretrained weights
            if 'pos_embed' in checkpoint.keys():  # Check if only backbone keys, directly import, need to check if pos_embed dimensions match
                if checkpoint['pos_embed'].shape != model_state_dict['pos_embed'].shape:
                    checkpoint['pos_embed'] = model_state_dict['pos_embed']
                self.model.backbone.feature_model.load_state_dict(checkpoint, strict=True)
            elif 'teacher' in checkpoint.keys():  # Indicates it's our pretrained model
                for key, values in checkpoint['teacher'].items():
                    if 'backbone' in key:
                        model_state_dict[key.replace('backbone.', '')] = values
                self.model.backbone.feature_model.load_state_dict(model_state_dict, strict=True)  # Import all backbone parts together
            else:
                raise ValueError("Invalid checkpoint")
            self.logger.info(f"Loaded pretrained model from {self.args.model.pretrained_weights}")
        except Exception as e:
            # Capture and print complete exception information
            self.logger.error(f"Failed to load pretrained model from {self.args.model.pretrained_weights}")
            self.logger.error(f"Exception: {e}")
            self.logger.error(traceback.format_exc())

    def train(self, n_epoch):
        # Switch model to training mode
        self.model.train()
        self.logger.info(f'Start training at epoch {n_epoch}/{self.total_epoch}')
        # Set tqdm progress bar to track training progress
        with tqdm(total=len(self.train_dataloader), desc=f"Train Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            epoch_loss = 0.0
            for i, data_dict in enumerate(self.train_dataloader):
                # Get images and labels
                if len(data_dict) == 5:
                    images, labels, extra_features, image_name, study_series_names = data_dict['image'], data_dict['label'], data_dict['feature'], data_dict['image_name'], data_dict['study_series_name']
                else:
                    raise ValueError(f"Invalid data_dict: {data_dict}")
                # Consider processing extra_features - e.g., change -1 to 0, add random label smoothing, swap -1 and 0, then add random noise as label smoothing
                # Swap -1 and 0
                if self.args.optim.bonecot_change01:
                    extra_features = extra_features + 1
                    extra_features[extra_features == 1] = 0
                    extra_features[extra_features == 2] = 1
                # Add random noise
                if self.args.optim.bonecot_label_smooth > 0:
                    # Generate random noise [-label_smooth, label_smooth]
                    random_noise = np.random.uniform(-self.args.optim.bonecot_label_smooth, self.args.optim.bonecot_label_smooth, size=extra_features.shape)
                    extra_features = extra_features + torch.tensor(random_noise)
                # Random mask
                if self.args.optim.bonecot_dropout > 0:
                    dropout_mask = torch.rand(extra_features.shape) > self.args.optim.bonecot_dropout
                    extra_features = extra_features * dropout_mask
                # Ensure type matches image
                extra_features = extra_features.to(images.dtype)
                # Merge image and feature tensor
                images = merge_bonecot_input_tensors(images, extra_features)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()

                # Automatic mixed precision support (optional)
                with self.amp_context:
                    outputs = self.model(images)
                    if self.args.optim.loss_type in ['BCE', 'BCE_Focal']:
                        labels = labels.float()
                        if outputs.shape != labels.shape and self.num_classes == 1:
                            labels = labels.unsqueeze(1)
                    else:
                        labels = labels.long()
                    loss = self.criterion(outputs, labels)

                # Update metric and loss
                self.metric.update(outputs.detach().cpu(), labels)
                epoch_loss += loss.item()

                if self.use_amp:
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Update parameters
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # Update real-time loss on progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            
                self.log_tensorboard.add_scalar('train_loss', loss.item(), n_epoch * len(self.train_dataloader) + i)

                # Stop training if exceeding maximum steps
                if self.max_step_per_epoch != -1 and i >= self.max_step_per_epoch:
                    break
                
            # Calculate average loss and metric results at the end of the epoch
            avg_loss = epoch_loss / len(self.train_dataloader)
            metrics_result = self.metric.results()
            for key, value in metrics_result.items():
                self.log_tensorboard.add_scalar(f'train_{key}', value, n_epoch)
            
            self.logger.info(f'Finish training epoch {n_epoch}')
            return avg_loss, metrics_result
        
    @torch.no_grad()
    def val(self, n_epoch, split='val'):
        # Add code to save prediction results - create a new dictionary for each val call
        val_results_dict = {} # Structure: barcode, sequence number, image_name, label, pred
        self.model.eval()
        dataloader = self.dataloaders_dict[split]
        self.logger.info(f'Start evaluate {split} epoch {n_epoch}')
        # Set tqdm progress bar to track evaluation progress
        with tqdm(total=len(dataloader), desc=f"{split} Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            epoch_loss = 0.0
            for i, data_dict in enumerate(dataloader):
                # Get images and labels
                if len(data_dict) == 5:
                    images, labels, extra_features, image_name, study_series_names = data_dict['image'], data_dict['label'], data_dict['feature'], data_dict['image_name'], data_dict['study_series_name']
                else:
                    raise ValueError(f"Invalid data_dict: {data_dict}")
                # Consider processing extra_features - e.g., change -1 to 0
                # Swap -1 and 0
                if self.args.optim.bonecot_change01:
                    extra_features = extra_features + 1
                    extra_features[extra_features == 1] = 0
                    extra_features[extra_features == 2] = 1
                    
                # Merge image and feature tensor
                images = merge_bonecot_input_tensors(images, extra_features)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Automatic mixed precision support (optional)
                with self.amp_context:  # Ensure self.use_amp is a boolean
                    outputs = self.model(images)
                    if self.args.optim.loss_type in ['BCE', 'BCE_Focal']:
                        labels = labels.float()
                        if outputs.shape != labels.shape and self.num_classes == 1:
                            labels = labels.unsqueeze(1)
                    else:
                        labels = labels.long()
                    loss = self.criterion(outputs, labels)
                    
                # Directly save output and labels using image_name and study_series_names as keys
                if len(data_dict) == 5:
                    for j in range(len(study_series_names)):
                        study_id, series_id = study_series_names[j].split('_')
                        if study_id not in val_results_dict.keys():
                            val_results_dict[study_id] = {}
                        if series_id not in val_results_dict[study_id].keys():
                            val_results_dict[study_id][series_id] = {}                        
                        val_results_dict[study_id][series_id][image_name[j]] = {'label': labels[j].cpu().numpy(), 'pred': outputs[j].cpu().numpy()} 
                        
                # Update metric and loss
                self.metric.update(outputs.detach().cpu(), labels)
                epoch_loss += loss.item()

                # Update real-time loss on progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            
            # Calculate average loss and metric results at the end of the epoch
            avg_loss = epoch_loss / len(dataloader)
            metrics_result = self.metric.results()
            for key, value in metrics_result.items():
                self.log_tensorboard.add_scalar(f'{split}_{key}', value, n_epoch)
                
            # Save prediction results
            save_file_name = os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch}.npz')
            np.savez(save_file_name, val_results_dict)
            self.logger.info(f"Save prediction results to {save_file_name}")
            self.logger.info(f'Finish evaluate {split} epoch {n_epoch}')
            return avg_loss, metrics_result
   
class BoneCoT_Finetune_Single_Linear_Trainer(BoneCoT_Finetune_Trainer):
    def __init__(self, args):
        super().__init__(args)
    
    def get_model(self):
        # Full parameter fine-tuning
        self.model, self.embed_dim = build_bonecot_single_linear_finetune_model_from_cfg(cfg=self.args, only_teacher=True)