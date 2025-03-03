from .finetune_linear_trainer import FinetuneLinearTrainer
from .o1_finetune_linear_trainer import O1FinetuneLinearTrainer
import torch






class EvalLinearTrainer(FinetuneLinearTrainer):
    def load_checkpoint(self):
        super().load_checkpoint()
        checkpoint_path = self.args.model.checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        self.model.load_state_dict(checkpoint['model']) 
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    def run(self):
        _, self.results_dict['val'][0] =  self.val(0, split='val')
        if self.args.data.val_dataset == self.args.data.test_dataset or self.args.data.test_dataset == '':
            self.results_dict['test'][0] = self.results_dict['val'][0]
        else:
            _, self.results_dict['test'][0] =  self.val(0, split='test')
        
        val_str = self.format_results(self.results_dict['val'][0])
        test_str = self.format_results(self.results_dict['test'][0])

        self.logger.info(f"Epoch {0}\nVal:\n{val_str}\nTest:\n{test_str}")
    
class EvalO1LinearTrainer(O1FinetuneLinearTrainer):
    def load_checkpoint(self):
        super().load_checkpoint()
        checkpoint_path = self.args.model.checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        self.model.load_state_dict(checkpoint['model'])
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
    def run(self):
        _, self.results_dict['val'][0] =  self.val(0, split='val')
        if self.args.data.val_dataset == self.args.data.test_dataset or self.args.data.test_dataset == '':
            self.results_dict['test'][0] = self.results_dict['val'][0]
        else:
            _, self.results_dict['test'][0] =  self.val(0, split='test')
        
        val_str = self.format_results(self.results_dict['val'][0])
        test_str = self.format_results(self.results_dict['test'][0])

        self.logger.info(f"Epoch {0}\nVal:\n{val_str}\nTest:\n{test_str}")
