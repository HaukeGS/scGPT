

### Imports
print("### Trying to import scGPT and other libraries...\n")

# systen relevant imports
import os
import sys
import argparse
import json
import time
from socket import gethostname
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

# torch relevant imports
import scanpy as sc
import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from datasets import Dataset, load_dataset, concatenate_datasets

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# scgpt imports
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.tokenizer import GeneVocab, random_mask_value
from scgpt.scbank import DataBank
from scgpt.utils import MainProcessOnly
from scgpt import logger

import argparser

print(f"### Imports successful!\n")



# %% SLURM setup and main
def setup_distributeddataparallel():
    """
    Setup the DistributedDataParallel (DDP) environment.
    """
    try:
        dist.init_process_group("nccl", rank=GLOBAL_RANK, world_size=WORLD_SIZE)
        if GLOBAL_RANK == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
        
        # Test communication with a barrier
        dist.barrier()
        printgpu(f"Successfully synchronized with all processes")
        
    except Exception as e:
        printgpu(f"Failed in setup: {e}")
        raise


def initialize_slurm_variables():
    """
    Initialize SLURM-related variables globally.
    """
    global NODE_ID, WORLD_SIZE, GPUS_PER_NODE, GLOBAL_RANK, LOCAL_RANK
    NODE_ID       = gethostname()
    WORLD_SIZE    = int(os.environ["WORLD_SIZE"])
    GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
    GLOBAL_RANK   = int(os.environ["SLURM_PROCID"])
    LOCAL_RANK = GLOBAL_RANK - GPUS_PER_NODE * (GLOBAL_RANK // GPUS_PER_NODE)
    torch.cuda.set_device(LOCAL_RANK)


def initialize_utility_variables(args: argparse.Namespace):
    """
    Initialize utility variables from the arguments globally.
    """
    global SAVE_DIR, USE_GENERATIVE_TRAINING, SPECIAL_TOKENS, USE_CLS, USE_CCE, MVC
    SAVE_DIR = Path(args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)
    USE_GENERATIVE_TRAINING = True if args.training_tasks in ["gen", "both"] else False
    SPECIAL_TOKENS = [args.pad_token, "<cls>", "<eoc>"]
    USE_CLS = not args.no_cls
    USE_CCE = not args.no_cce
    MVC = True
    if GLOBAL_RANK == 0:
        with open(SAVE_DIR / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    scg.utils.set_seed(42)


def get_arguments():
    """
    Get command line arguments.  
    Arguments are specified in the argparser.py file.
    """
    parser = argparser.get_parser()
    return parser.parse_args()


def printgpu(string: str, flush: bool = True):
    """
    Print a message with infos about which GPU is being used.
    """
    output = f"[GPU:{GLOBAL_RANK}][{NODE_ID}:{LOCAL_RANK}] {string}"
    print(output, flush=flush)


def main():
    initialize_slurm_variables()
    args = get_arguments()
    initialize_utility_variables(args)
    print(f"NODEID: {NODE_ID}, GLOBAL_RANK: {GLOBAL_RANK}, WORLD_SIZE: {WORLD_SIZE}, SLURM_GPUS_ON_NODE: {GPUS_PER_NODE}, LOCAL_RANK: {LOCAL_RANK}, DEVICE_COUNT: {torch.cuda.device_count()}", flush=True)

    setup_distributeddataparallel()
    try:
        printgpu(f"Setup complete")
        dist.barrier()
        time.sleep(2)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()


# %% Data preprocessing




# %% Model initialization




# %% Training and evaluation