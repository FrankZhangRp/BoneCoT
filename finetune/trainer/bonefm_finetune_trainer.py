from .base_trainer import BaseTrainer
from ..models import build_bonefm_model_from_cfg
import torch
import traceback

class BoneFM_Finetune_Trainer(BaseTrainer):
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
        self.model, self.embed_dim = build_bonefm_model_from_cfg(cfg=self.args, only_teacher=True)
        
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
            