import torch
from .base import Hook

class ModelInitHook(Hook):
    def before_run(self, runner):
        # Fix pretraining weights 
        if runner.cfg.model.get('fix_pretrain', None):
            if runner.local_rank == 0:
                runner.logger.info("Fix pretrain-model weights")
            runner.fix_pretrain_weights()
        # Load training resume
        resume_path = runner.cfg.Resume
        if resume_path is not None and runner.local_rank == 0:
            runner.load_checkpoint(resume_path)
            runner.logger.info("Resume from {}".format(resume_path))

