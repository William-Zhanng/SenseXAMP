from .base import Hook

class BufferHook(Hook):
    def after_val_epoch(self, runner):
        runner.log_buffer.clear_all()
    