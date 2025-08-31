

### Imports
print("### Trying to import scGPT and other libraries...")

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

print(f"### Imports successful!")


# %% Data preprocessing

def get_vocabulary(vocab_path: Path) -> GeneVocab:
    vocab = GeneVocab.from_file(vocab_path)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    return vocab

# def get_dataset(data_source_path: Path, vocab: GeneVocab) -> Dataset:
#     if (data_source_path.is_dir()):

    # print(f"\n### Loading data from {data_source_path}...\n")
    # # collection of parquet files
    # parquet_files = [str(f) for f in Path(args.data_source).glob("*.parquet")]
    # print(f"\n### Found {len(parquet_files)} parquet files in {args.data_source}")
    # cache_dir = Path(args.data_source).parent / "cache"
    # vocab = GeneVocab.from_file(Path(args.vocab_path))
    # for s in special_tokens:
    #     if s not in vocab:
    #         vocab.append_token(s)
    # if USE_CCE or USE_CLS or MVC:
    #     print(f"\n### Loading data with <cls> prefix from {args.data_source}...\n")
    #     # load or make the dataset w/ <cls> appended at the beginning
    #     cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
    #     if not cls_prefix_datatable.exists():
    #         if args.local_rank in [0, -1]:
    #             logger.info(f"Rank {args.local_rank}: Preparing dataset")
    #             raw_dataset = load_dataset(
    #                 "parquet",
    #                 data_files=parquet_files,
    #                 split="train",
    #                 cache_dir=str(cache_dir),
    #             )
    #             raw_dataset = _map_append_cls(raw_dataset)
    #             raw_dataset.to_parquet(str(cls_prefix_datatable))
    #         if IS_DATA_PARALLEL:
    #             torch.distributed.barrier()  # wait for the mapping to finish
    #     raw_dataset = load_dataset(
    #         "parquet",
    #         data_files=str(cls_prefix_datatable),
    #         split="train",
    #         cache_dir=str(cache_dir),
    #     )
    #     logger.info(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")


# %% Model initialization




# %% Training and evaluation



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
    scg.utils.add_file_handler(logger, SAVE_DIR / "run.log")


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
        vocab = get_vocabulary(Path(args.vocab_path))
        print(f"Data Sources: {args.data_source}")
        dist.barrier()
        time.sleep(2)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()