

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


### SLURM setup and main


def setup_distributeddataparallel():
    # initialize the process group
    dist.init_process_group("nccl", rank=GLOBAL_RANK, world_size=WORLD_SIZE)
    if GLOBAL_RANK == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)


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


def get_arguments():
    """
    Get command line arguments.  
    Arguments are specified in the argparser.py file.
    """
    parser = argparser.get_parser()
    return parser.parse_args()


def main():
    initialize_slurm_variables()
    print(f"NODEID: {NODE_ID}, GLOBAL_RANK: {GLOBAL_RANK}, WORLD_SIZE: {WORLD_SIZE}, SLURM_GPUS_ON_NODE: {GPUS_PER_NODE}, LOCAL_RANK: {LOCAL_RANK}", flush=True)
    assert GPUS_PER_NODE == torch.cuda.device_count()


    args = get_arguments()
    print(f"Arguments:\n {json.dumps(vars(args), indent=2)}\n")
    setup_distributeddataparallel()
    try:
        print(f"Setup complete")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()