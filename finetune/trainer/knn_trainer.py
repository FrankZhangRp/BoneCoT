import torch
import torch.utils
from models import build_knn_model_from_cfg
from tqdm import tqdm
import numpy as np
import traceback
from .base_trainer import BaseTrainer

class KNNTrainer(BaseTrainer):
    def __init__(self, args):
        super(KNNTrainer, self).__init__(args)
        self.knn_k = args.model.knn_k
        self.total_epoch = 1
        self.use_amp = self.args.optim.use_amp
    
    def knn_init(self):
        self.features_list = []
        self.labels_list = []
    
    def get_model(self):
        self.model, self.embed_dim = build_knn_model_from_cfg(cfg=self.args)

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.args.model.pretrained_weights}")
        if self.args.model.pretrained_weights == "":
            self.logger.info("No checkpoint loaded.")
            return
        try:
            model_state_dict = self.model.backbone.state_dict()
            checkpoint = torch.load(self.args.model.pretrained_weights, map_location='cpu', weights_only=True)
            if 'pos_embed' in checkpoint.keys():
                if checkpoint['pos_embed'].shape != model_state_dict['pos_embed'].shape:
                    checkpoint['pos_embed'] = model_state_dict['pos_embed']
                self.model.backbone.load_state_dict(checkpoint, strict=True)
            elif 'teacher' in checkpoint.keys():
                self.model.load_state_dict(checkpoint['teacher'], strict=False)
            else:
                raise ValueError("Invalid checkpoint")
            self.logger.info(f"Loaded pretrained model from {self.args.model.pretrained_weights}")
        except Exception as e:
            self.logger.error(f"Failed to load pretrained model from {self.args.model.pretrained_weights}")
            self.logger.error(f"Exception: {e}")
            self.logger.error(traceback.format_exc())
            
    @torch.no_grad()
    def train(self, n_epoch):
        self.model.train()
        self.logger.info(f"Start training epoch {n_epoch}")
        with tqdm(total=len(self.train_dataloader), desc=f"Train Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            for i, data_dict in enumerate(self.train_dataloader):
                images, labels = data_dict['image'], data_dict['label']
                images = images.to(self.device)
                with self.amp_context:
                    outputs = self.model(images)
                self.features_list.append(outputs.detach().cpu().numpy())
                self.labels_list.append(labels.numpy())

                pbar.set_postfix(loss=0.)
                pbar.update(1)
        if self.dataparallel:
            self.model.module.fit_knn(np.concatenate(self.features_list, axis=0), np.concatenate(self.labels_list, axis=0))
        else:
            self.model.fit_knn(np.concatenate(self.features_list, axis=0), np.concatenate(self.labels_list, axis=0))
        self.logger.info(f"KNN model fitted with {self.args.model.knn_k} neighbors")
        return 0, {}
    
    @torch.no_grad()
    def val(self, n_epoch, split='val'):
        self.model.eval()
        dataloader = self.dataloaders_dict[split]
        self.logger.info(f"Start validation on {split} set")
        with tqdm(total=len(dataloader), desc=f"{split} Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            for i, data_dict in enumerate(dataloader):
                images, labels = data_dict['image'], data_dict['label']
                images = images.to(self.device)
                with self.amp_context:
                    preds = self.model(images)
                if self.args.optim.loss_type == 'CE':
                    self.metric.update(preds, labels, easy_model=True)
                else:
                    self.metric.update(preds, labels)
                pbar.update(1)
        results_dict = self.metric.results()
        self.logger.info(f"{split} results: {results_dict}")
        return 0, results_dict
    
    def run(self):
        self.knn_init()
        self.train(0)
        _, self.results_dict['val'] = self.val(0, split='val')
        log_str = " / ".join([f"{key}: {value:.4f}" for key, value in self.results_dict['val'].items()])
        self.logger.info(f"Val: {log_str}")
        if self.args.data.test_dataset is not None and self.args.data.test_dataset != self.args.data.val_dataset:
            _, self.results_dict['test'] = self.val(0, split='test')
        else:
            self.results_dict['test'] = self.results_dict['val']
        log_str = " / ".join([f"{key}: {value:.4f}" for key, value in self.results_dict['test'].items()])
        self.logger.info(f"Test: {log_str}")
        
        self.save_checkpoint(0)