from .bonefm_finetune_trainer import BoneFM_Finetune_Trainer
import torch
from torchvision import transforms
from ..data import SeriesClassificationDataset
class BoneFM_Eval_Trainer(BoneFM_Finetune_Trainer):
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
        self.train_dataset = SeriesClassificationDataset(self.args.data.train_dataset, transform=self.train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        assert self.args.data.val_dataset is not None
        self.val_dataset = SeriesClassificationDataset(self.args.data.val_dataset, transform=self.test_transform)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        if self.args.data.test_dataset is not None:
            self.test_dataset = SeriesClassificationDataset(self.args.data.test_dataset, transform=self.test_transform)
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
        
        self.logger.info(f"Test dataset: {self.args.data.test_dataset} / Length: {len(self.test_dataset)} / loader: {len(self.test_dataloader)}")
    
    
    def load_checkpoint(self):
        super().load_checkpoint()
        backbone_ckpt_path = self.args.model.backbone_ckpt_path
        checkpoint = torch.load(self.args.model.backbone_ckpt_path, map_location='cpu', weights_only=True)
        model_state_dict = self.model.backbone.feature_model.state_dict()
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
        classifier_ckpt_path = self.args.model.classifier_ckpt_path
        classifier_checkpoint = torch.load(classifier_ckpt_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(classifier_checkpoint, strict=False) 
        self.logger.info(f"Loading backbone from {backbone_ckpt_path} and classifier from {classifier_ckpt_path}")
    
    def run(self):
        _, self.results_dict['test'][0] =  self.val(0, split='test')