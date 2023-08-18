from .base import Hook
class LoggerHook(Hook):
    def __init__(self, batch_interval, output_training_vars = True) -> None:
        self.batch_interval = batch_interval
        self.output_training_vars = output_training_vars

    def before_train_epoch(self, runner):
        if runner.local_rank == 0:
            if self.output_training_vars:
                runner.logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Time Elapsed':^9}")
            else:
                runner.logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Time Elapsed':^9}")
    
    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner,self.batch_interval):
            if runner.local_rank == 0:
                cur_epoch = runner.epoch+1
                cur_iter = runner.inner_iter+1
                buffer_info = runner.log_buffer.output_results(self.batch_interval)
                time_elapsed = buffer_info['batch_time']
                if self.output_training_vars:
                    batch_loss = buffer_info['loss']
                    runner.logger.info(
                        f"{cur_epoch:^7} | {cur_iter:^7} | {batch_loss:^12.6f} | {time_elapsed:^9.2f}")
                else:
                    runner.logger.info(
                        f"{cur_epoch:^7} | {cur_iter:^7} | {time_elapsed:^9.2f}")

    def before_val_epoch(self, runner):
        if runner.local_rank == 0:
            cur_epoch = runner.epoch+1
            num_val = len(runner.cur_dataloader)
            runner.logger.info(
                "epoch {} start evaluation! Total {} batches to evaluation.\n".format(cur_epoch,num_val))

    def after_val_epoch(self, runner):
        """
        输出val的结果
        """
        if runner.local_rank == 0:
            cur_epoch = runner.epoch+1
            cur_res = runner.cur_val_results
            time_elapsed = runner.epoch_elapsed
            runner.logger.info(
                "epoch {} time elapsed: {}".format(cur_epoch,time_elapsed)
            )
            runner.logger.info(
                "evaluation results:\n{}".format(cur_res))
            runner.logger.info('Current best results:\n {}'.format(runner.best_val_results))