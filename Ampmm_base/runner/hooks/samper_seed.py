from .base import Hook
class DistributedHook(Hook):
    def before_epoch(self, runner):
        data_loader = runner.train_dataloader
        data_loader.sampler.set_epoch(runner.epoch)