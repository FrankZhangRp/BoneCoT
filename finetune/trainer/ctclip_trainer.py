from .base_trainer import BaseTrainer
from models import build_ct_clip_finetune_model_from_cfg
import torch
from torchvision import transforms
from data import Series3DClassificationDataset

class CTCLIP_Trainer(BaseTrainer):
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
        self.model, self.embed_dim = build_ct_clip_finetune_model_from_cfg(self.args)
    
    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.args.model.pretrained_weights}")
        checkpoint_path = self.args.model.pretrained_weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        selected_model_dict = {}
        for key_name in checkpoint.keys():
            if 'visual_transformer' in key_name:
                selected_model_dict[key_name.replace('visual_transformer.', '')] = checkpoint[key_name]
            if 'to_visual_latent' in key_name:
                selected_model_dict[key_name.replace('trained_model.s', '')] = checkpoint[key_name]
        self.model.load_state_dict(selected_model_dict, strict=False)
        self.logger.info(f"Checkpoint {self.args.model.pretrained_weights} loaded.")
    
    def get_data(self):
        self.train_transform = None
        
        self.test_transform = None
        
        assert self.args.data.train_dataset is not None
        self.train_dataset = Series3DClassificationDataset(self.args.data.train_dataset, transform=self.train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        assert self.args.data.val_dataset is not None
        self.val_dataset = Series3DClassificationDataset(self.args.data.val_dataset, transform=self.test_transform)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        if self.args.data.test_dataset is not None:
            self.test_dataset = Series3DClassificationDataset(self.args.data.test_dataset, transform=self.test_transform)
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
    
    def get_optimizer(self):
        learning_rate = self.args.optim.lr
        weight_decay = self.args.optim.weight_decay
        if self.args.optim.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(self.args.optim.adamw_beta1, self.args.optim.adamw_beta2))
        elif self.args.optim.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(self.args.optim.adamw_beta1, self.args.optim.adamw_beta2))
        elif self.args.optim.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=self.args.optim.sgd_momentum)
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optim.optimizer}")
    

class CTCLIP_Eval_Trainer(CTCLIP_Trainer):
    def load_checkpoint(self):
        if self.args.model.checkpoint_path is not None:
            self.logger.info(f"Loading checkpoint from {self.args.model.checkpoint_path}")
            checkpoint = torch.load(self.args.model.checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.logger.info(f"Checkpoint {self.args.model.checkpoint_path} loaded.")
        else:
            self.logger.info('No checkpoint path provided, using pretrained weights.')
            
    def run(self):
        n_epoch = 0
        _, self.results_dict['val'][n_epoch] =  self.val(n_epoch, split='val')
        if self.args.data.val_dataset == self.args.data.test_dataset or self.args.data.test_dataset == '':
            self.results_dict['test'][n_epoch] = self.results_dict['val'][n_epoch]
        else:
            _, self.results_dict['test'][n_epoch] =  self.val(n_epoch, split='test')
        
        val_str = self.format_results(self.results_dict['val'][n_epoch])
        test_str = self.format_results(self.results_dict['test'][n_epoch])

        self.logger.info(f"Epoch {n_epoch}\nVal:\n{val_str}\nTest:\n{test_str}")
