

### Imports

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



# %% Utility functions

def is_master_gpu():
    return GLOBAL_RANK == 0


def printmaster(string: str, flush: bool = True):
    if is_master_gpu():
        print(f"[MASTER] {string}", flush=flush)


def printgpu(string: str, flush: bool = True):
    """
    Print a message with infos about which GPU is being used.
    """
    output = f"[GPU:{GLOBAL_RANK}][{NODE_ID}:{LOCAL_RANK}] {string}"
    print(output, flush=flush)


def dump_args(args: argparse.Namespace) -> None:
    """
    Dump the arguments to a json file in the save directory.
    """
    if is_master_gpu():
        with open(SAVE_DIR / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)


# %% Data preprocessing

def get_vocabulary(vocab_path: Path) -> GeneVocab:
    """
    Load the vocabulary from the specified path.
    If the special tokens are not in the vocabulary, add them.
    """
    vocab = GeneVocab.from_file(vocab_path)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    return vocab


def dump_vocab(vocab: GeneVocab) -> None:
    """
    Dump the vocabulary to a json file in the save directory.
    """
    if is_master_gpu():
        with open(SAVE_DIR / "vocab.json", "w") as f:
            json.dump(
                {token: index for token, index in vocab.get_stoi().items()},
                f,
                indent=2,
        )
            

def create_dataset(data_paths: List[Path]):
    """
    Load the dataset from the specified data source.
    data_source can contain multiple directories. See argparser.py for more details
    This assumes one or more directories with a preprocessed cls_prefix_data.parquet file. 
    Returns the concatenated datasets from all directories.
    """
    if not data_paths:
        raise ValueError("No data paths provided")
    datasets = []
    example_count = 0
    for path in data_paths:
        cls_prefix_datatable = path / "cls_prefix_data.parquet"
        if not cls_prefix_datatable.exists():
            raise FileNotFoundError(f"File not found: {cls_prefix_datatable}")

        raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
            cache_dir=str(path.parent / "cache"),
        )
        datasets.append(raw_dataset)
        printmaster(f"Dataset type: {type(raw_dataset)}")
        printmaster(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
        example_count += len(raw_dataset)
    printmaster(f"Total examples across all datasets: {example_count}")
    # needs raw_dataset_merging
    merged_dataset = concatenate_datasets(datasets)
    return merged_dataset


def create_collator(
        vocab: GeneVocab, 
        max_seq_len: int, 
        pad_token: str, 
        pad_value: int,
        input_style: str,
        mask_ratio: Union[float, List[float]],
        mask_value: float,
        trunc_by_sample: bool,
        training_tasks: str,
    ) -> scg.DataCollator:
    """
    Create a data collator for the dataloader with provided arguments.
    """
    collator = scg.DataCollator(
        do_padding=True if max_seq_len is not None else False,
        pad_token_id=vocab[pad_token],
        pad_value=pad_value,
        do_mlm=True,
        do_binning=True if input_style == "binned" else False,
        mlm_probability=mask_ratio,
        mask_value=mask_value,
        max_length=max_seq_len,
        sampling=trunc_by_sample,
        data_style=training_tasks,
    )
    return collator


def create_dataloaders(train_dataset: Dataset, validation_dataset: Dataset, batch_size: int, collator: scg.DataCollator) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for training and validation datasets.
    Uses DistributedSampler for the training loader and uses the full validation dataset for the validation loader.
    """
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True)
    else:
        raise ValueError("Distributed not initialized")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=CPUS_PER_TASK,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        # prefetch_factor=CPUS_PER_TASK
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=CPUS_PER_TASK,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
        # prefetch_factor=CPUS_PER_TASK
    )
    return train_loader, validation_loader


def get_dataloaders(
        data_paths: List[Path], 
        validation_ratio: float, 
        vocab: GeneVocab,
        max_seq_len: int, 
        pad_token: str, 
        pad_value: int,
        input_style: str,
        mask_ratio: Union[float, List[float]],
        mask_value: float,
        trunc_by_sample: bool,
        training_tasks: str,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for training and validation datasets.
    """
    dataset = create_dataset(data_paths)
    printmaster(f"Final dataset length: {len(dataset)}")
    printmaster(f"Dataset format: {dataset.format}")
    collator = create_collator(
        vocab=vocab,
        max_seq_len=max_seq_len,
        pad_token=pad_token,
        pad_value=pad_value,
        input_style=input_style,
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        trunc_by_sample=trunc_by_sample,
        training_tasks=training_tasks,
    )
    train_dataset, validation_dataset = dataset.train_test_split(test_size=validation_ratio, shuffle=True, seed=SEED).values()
    train_loader, validation_loader = create_dataloaders(train_dataset, validation_dataset, batch_size, collator)
    dist.barrier()
    time.sleep(2)
    printmaster(f"Train loader length: {len(train_loader)}")
    printmaster(f"Validation loader length: {len(validation_loader)}")
    return train_loader, validation_loader



# %% Model initialization




# %% Training and evaluation



# %% setup and initialization


def setup_distributeddataparallel():
    """
    Setup the DistributedDataParallel (DDP) environment.
    """
    try:
        dist.init_process_group("nccl", rank=GLOBAL_RANK, world_size=WORLD_SIZE)
        
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
    global NODE_ID, WORLD_SIZE, GPUS_PER_NODE, GLOBAL_RANK, LOCAL_RANK, CPUS_PER_TASK
    NODE_ID       = gethostname()
    WORLD_SIZE    = int(os.environ["WORLD_SIZE"])
    GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
    GLOBAL_RANK   = int(os.environ["SLURM_PROCID"])
    CPUS_PER_TASK = int(os.environ["SLURM_CPUS_PER_TASK"])
    LOCAL_RANK = GLOBAL_RANK - GPUS_PER_NODE * (GLOBAL_RANK // GPUS_PER_NODE)
    printgpu(f"CPUS_PER_TASK: {CPUS_PER_TASK}")
    torch.cuda.set_device(LOCAL_RANK)


def initialize_utility_variables(args: argparse.Namespace):
    """
    Initialize utility variables from the arguments globally.
    """
    global SAVE_DIR, USE_GENERATIVE_TRAINING, SPECIAL_TOKENS, SEED
    # global USE_CLS, USE_CCE, MVC
    SAVE_DIR = Path(args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)
    USE_GENERATIVE_TRAINING = True if args.training_tasks in ["gen", "both"] else False
    SPECIAL_TOKENS = [args.pad_token, "<cls>", "<eoc>"]
    # USE_CLS = not args.no_cls
    # USE_CCE = not args.no_cce
    # MVC = True

    SEED = 42

    scg.utils.set_seed(SEED)
    scg.utils.add_file_handler(logger, SAVE_DIR / "run.log")


def initialize_additional_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """
    Initialize additional arguments based on the provided args.
    """
    if args.input_emb_style == "category":
        args.mask_value = args.n_bins + 1
        args.pad_value = args.n_bins  # for padding gene expr values
        n_input_bins = args.n_bins + 2
    else:
        args.mask_value = -1
        args.pad_value = -2
        n_input_bins = args.n_bins

    if args.training_tasks in ["gen", "both"]:
        args.mask_ratio = [0.25, 0.50, 0.75]
    return args


def get_arguments():
    """
    Get command line arguments.  
    Arguments are specified in the argparser.py file.
    """
    parser = argparser.get_parser()
    return parser.parse_args()


# %% main


def main():
    initialize_slurm_variables()
    args = get_arguments()
    args = initialize_additional_arguments(args)
    initialize_utility_variables(args)
    dump_args(args)
    print(f"NODEID: {NODE_ID}, GLOBAL_RANK: {GLOBAL_RANK}, WORLD_SIZE: {WORLD_SIZE}, SLURM_GPUS_ON_NODE: {GPUS_PER_NODE}, LOCAL_RANK: {LOCAL_RANK}, DEVICE_COUNT: {torch.cuda.device_count()}", flush=True)

    setup_distributeddataparallel()
    printmaster(f"Setup complete. Group initialized? {dist.is_initialized()}", flush=True)
    try:
        vocab = get_vocabulary(Path(args.vocab_path))
        dump_vocab(vocab)
        train_loader, validation_loader = get_dataloaders(
            data_paths=argparser.get_datapaths(args), 
            validation_ratio=args.valid_size_or_ratio, 
            vocab=vocab,
            max_seq_len=args.max_seq_len,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            input_style=args.input_emb_style,
            mask_ratio=args.mask_ratio,
            mask_value=args.mask_value,
            trunc_by_sample=args.trunc_by_sample,
            training_tasks=args.training_tasks,
            batch_size=args.batch_size,
        )
        dist.barrier()
        time.sleep(2)
        model = get_model()
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        printgpu(f"Exiting...")

if __name__ == "__main__":
    main()