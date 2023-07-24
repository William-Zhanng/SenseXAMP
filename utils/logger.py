import logging
import os
import time

def Logger(out_dir, fname='/log.txt'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)
    fHandler = logging.FileHandler(out_dir + fname, mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)

    logger.addHandler(fHandler)

    return logger
