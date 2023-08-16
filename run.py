from ast import arguments
import os
import sys
import time
import random
import argparse
import torch
import numpy as np
# For DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import Config,Logger
from Ampmm_base.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Training SenseXAMP benchmark')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--mode', default='train', help='train or test')
    # for ddp
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args

def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# # for ddp
# parser.add_argument("--local_rank", default=-1, type=int)
if __name__ == '__main__':
    args = parse_args()
    local_rank = args.local_rank
    # ddp init
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    # set random seed
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    # create logger and work_dir
    if dist.get_rank() == 0:
        cfg.work_dir = os.path.join(cfg.work_dir,cfg.benchmark_name,cfg.dataset_name,args.mode,
                                    time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(cfg.work_dir,exist_ok=True)
    logger = Logger(cfg.work_dir)
    if args.local_rank == 0:
        logger.info(args)
        logger.info("Running with config:\n{}".format(cfg.text))
        # set random seeds
        if args.seed is not None:
            logger.info('Set random seed to {}'.format(args.seed))
            set_seed(args.seed)
    runner = Runner(cfg,logger,args.local_rank,args.mode)
    if args.mode == 'train':
        runner.run()
    elif args.mode == 'test':
        runner.test()
    else:
        if args.local_rank == 0:
            print("Please ensure args.mode to be train or test")
            exit()
