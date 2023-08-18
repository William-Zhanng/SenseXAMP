import os
import os.path as osp
from .hooks import *
from .base_runner import BaseRunner
from .api import default_batch_processor,ranking_batch_processor,base_batch_processor
from .optimizer import build_optimizer
from .log_buffer import LogBuffer
from Ampmm_base.data.datasets import build_dataset
from Ampmm_base.data.dataloaders import build_data_loader
from Ampmm_base.models import build_model
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import optimization
from tqdm import tqdm
# for ddp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torch

class Runner(BaseRunner):
    """
    An epoch based trainer
    """
    def __init__(self, cfg, logger, local_rank, mode) -> None:
        super(Runner,self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.benchmark_name = cfg.benchmark_name
        self.local_rank = local_rank
        self.max_epochs = cfg.epochs
        self.build_dataloaders(cfg)
        self.logger = logger
        if local_rank == 0:
            self.logger.info("Successfully create dataloaders")
        self.model = build_model(cfg)
        self.model.to(local_rank)
        if local_rank == 0:
            self.logger.info("Successfully create model: {}".format(self.model.__class__.__name__ ))
        self.get_batch_processor()
        self.log_buffer = LogBuffer()
        # batch_converter for data 
        self.register_default_hooks()

    def build_dataloaders(self, cfg):
        # Amp ranking task use weighted sampler
        weighted = True if (cfg.benchmark_name == 'amp_ranking') or (cfg.get('use_weighted_sampler',False)) else False
        if self.mode == 'train':
            self.train_dataset = build_dataset(cfg,'train')
       
            self.val_dataset = build_dataset(cfg,'val')

            self.train_dataloader = build_data_loader(
                self.train_dataset,cfg.data['train'].batch_per_gpu,'train',weighted
            )

            self.val_dataloader = build_data_loader(
                self.val_dataset,cfg.data['val'].batch_per_gpu,'val',weighted
            )
        
        else:
            self.test_dataset = build_dataset(cfg,'test')

            self.test_dataloader = build_data_loader(
                self.test_dataset,cfg.data['test'].batch_per_gpu,'val',weighted
            )

    def get_batch_processor(self):
        esm_base_models = ['EsmBaseClsModel','EsmBaseMultiLabelClsModel','EsmBaseRegModel',]
        esm_base_ranking_models = ['EsmBaseRankingModel']
        base_models = ['BaseClsModel','BaseFusionClsModel','BaseStcClsModel']
        if self.cfg.model['name'] in esm_base_models:
            self.batch_processor = default_batch_processor
            self.logger.info('runner got default_batch_processor')
        elif self.cfg.model['name'] in esm_base_ranking_models:
            self.batch_processor = ranking_batch_processor
            self.logger.info('runner got ranking_batch_processor')
        # elif self.cfg.model['name'] in base_models:
        else:
            self.batch_processor = base_batch_processor
            self.logger.info('runner got base_batch_processor')
            
    def register_default_hooks(self):
        batch_interval = self.cfg.batch_interval
        output_training_vars = True
        if self.cfg.benchmark_name == 'amp_ranking':
            output_training_vars = False
        self.register_hook(IterTimeHook(batch_interval=batch_interval),50)
        self.register_hook(LoggerHook(batch_interval,output_training_vars),60)
        self.register_hook(BufferHook(),90)
    
    def register_training_hooks(self):
        """
        Register default and custom hooks for training.
        """
        opt_cfg = self.cfg.optimizer_config
        self.register_hook(ModelInitHook(),0)
        self.register_hook(DistributedHook(),10)
        self.register_hook(OptimizerHook(opt_cfg),10)

    def register_evaluation_hooks(self):
        """
        Register default and custom hooks for evaluation.
        """
        if self.cfg.benchmark_name == 'amp_cls':
            cls_threshold = self.cfg.get('cls_threshold', 0.5)
            self.logger.info('The classfication threshold is: {}'.format(cls_threshold))
            self.register_hook(AmpClsEvalHook(cls_threshold),10)
        elif self.cfg.benchmark_name == 'amp_multilabel_cls':
            self.register_hook(AmpMultiLabelClsEvalHook(),10)
        elif self.cfg.benchmark_name == 'amp_ranking':
            self.register_hook(AmpRankingEvalHook(),10)
        elif self.cfg.benchmark_name == 'amp_reg':
            self.register_hook(AmpRegEvalHook(),10)

    def register_test_hooks(self):
        """
        Register hooks for analysing results of test.
        """
        if self.cfg.benchmark_name == 'amp_cls':
            cls_threshold = self.cfg.get('cls_threshold', 0.5)
            self.logger.info('The classfication threshold is: {}'.format(cls_threshold))
            self.register_hook(AmpClsAnalyseHook(self.cfg.work_dir,cls_threshold),10)
        elif self.cfg.benchmark_name == 'amp_multilabel_cls':
            self.register_hook(AmpMultiLabelClsAnalyseHook(self.cfg.work_dir),10)
        elif self.cfg.benchmark_name == 'amp_ranking':
            self.register_hook(AmpRankingAnalyseHook(self.cfg.work_dir),10)
        elif self.cfg.benchmark_name == 'amp_reg':
            self.register_hook(AmpRegAnalyseHook(self.cfg.work_dir),10)
            
    def fix_pretrain_weights(self):
        fixed_params = ["layers.{}".format(i) for i in range(32)]

        def trainable(model_params):
            for param in fixed_params:
                if param in model_params:
                    return False
            return True

        for name, value in self.model.named_parameters():
            if trainable(name):
                value.requires_grad = True
            else:
                value.requires_grad = False

    def load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))

    def save_checkpoint(self):
        self.logger.info('Got better results on validation set, model checkpoint updated')
        model_path = os.path.join(self.cfg.work_dir,"final.ckpt")
        torch.save(self.model.module.state_dict(), model_path)

    def run_iter(self, data_batch, train_mode):
        prots = data_batch['seq']
        outputs = self.batch_processor(
            self.model,data_batch,self.local_rank,train_mode)
        # train mode
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'],len(prots))
        self.outputs = outputs

    def train(self):
        """
        Train an epoch
        """
        self.model.train()
        self.mode = 'train'
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.train_dataloader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1
        self.call_hook('after_train_epoch')

    @torch.no_grad()
    def val(self, dataloader):
        self.model.eval()
        self.mode = 'val'
        self.cur_dataloader = dataloader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in tqdm(enumerate(dataloader)):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
        self.call_hook('after_val_epoch')    

    def run(self):
        """
        Used for training a model
        """
        self.register_training_hooks()
        self.register_evaluation_hooks()
        if self.local_rank == 0:
            self.logger.info('Start running,workdir:{}'.format(self.cfg.work_dir))

        self.call_hook('before_run')
        self.model = DDP(
            self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
            )
        self.optimizer = build_optimizer(self.model, self.cfg.optimizer)
        total_steps = (self.max_epochs) * len(self.train_dataloader)
        self.scheduler = optimization.get_cosine_schedule_with_warmup(self.optimizer,
                                                                    num_warmup_steps=500,  # Default value
                                                                    num_training_steps=total_steps)
        if self.local_rank == 0:
            self.logger.info("Start training...\n")

        while self.epoch < self.max_epochs:
            self.train()
            if self.local_rank == 0:
                self.val(self.val_dataloader)
            self._epoch += 1
            
    @torch.no_grad()
    def test(self):
        """
        Used for evaluation on test dataset
        """
        self.register_test_hooks()
        if self.local_rank == 0:
            self.logger.info('Start running,workdir:{}'.format(self.cfg.work_dir))
        self.model.eval()
        self.cur_dataloader = self.test_dataloader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in tqdm(enumerate(self.cur_dataloader)):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
        self.call_hook('after_val_epoch') 