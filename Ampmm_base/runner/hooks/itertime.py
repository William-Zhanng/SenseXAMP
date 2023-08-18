import time
from .base import Hook
class IterTimeHook(Hook):
    """
    Calculate time elapse during training/evaluation
    """
    def __init__(self,batch_interval):
        self.batch_interval = batch_interval
        
    def before_epoch(self, runner):
        self.epoch_time = time.time()    # when epoch start
        self.batch_time = time.time()    # when batch start

    def after_iter(self, runner):
        runner.log_buffer.update({'batch_time':time.time()-self.batch_time})
        self.batch_time = time.time()
    
    def after_val_epoch(self, runner):
        runner.epoch_elapsed = time.time()-self.epoch_time