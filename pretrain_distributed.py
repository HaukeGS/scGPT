

### Imports

# systen relevant imports
import math
import os
import sys
import argparse
import json
import time
import random
from socket import gethostname
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

# torch relevant imports
import scanpy as sc
import numpy as np
import torch
import transformers
import hashlib
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from datasets import Dataset, load_dataset, concatenate_datasets, interleave_datasets, IterableDataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# scgpt imports
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.streamingdataloaderwrapper import StreamingDataLoaderWrapper
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.tokenizer import GeneVocab, random_mask_value
from scgpt.scbank import DataBank
from scgpt.utils import MainProcessOnly
from scgpt import logger

import argparser



# %% Utility functions

def is_master_gpu():
    return GLOBAL_RANK == 0


def printtime(string: str, flush: bool = True):
    current_runtime = time.time() - runtime_startTime
    current_runtime = str(timedelta(seconds=int(current_runtime)))
    print(f"[{current_runtime}]|{string}", flush=flush)


def printmaster(string: str, flush: bool = True):
    if is_master_gpu():
        printtime(f"[MASTER] {string}", flush=flush)


def printlogging(string: str, level: str, flush: bool = True):
    if is_master_gpu():
        printtime(f"[{level.upper()}] {string}", flush=flush)


def printgpu(string: str, flush: bool = True):
    """
    Print a message with infos about which GPU is being used.
    """
    output = f"[GPU:{GLOBAL_RANK}][{NODE_ID}:{LOCAL_RANK}] {string}"
    printtime(output, flush=flush)


def dump_args(args: argparse.Namespace) -> None:
    """
    Dump the arguments to a json file in the save directory.
    """
    if is_master_gpu():
        with open(SAVE_DIR / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    printmaster(json.dumps(vars(args), indent=2))


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
            

def dump_debug(output: str) -> None:
    """
    Dump the debug information to a text file in the save directory.
    """
    if is_master_gpu():
        with open(SAVE_DIR / "debug.txt", "a") as f:
            f.write(output + "\n")


def get_tissue_sample_count(tissue: str) -> int:
    """
    Get the number of samples in the specified tissue.
    """
    with open("/home/hauke.schuele/scGPT_distributed/total_dataset_sample_counts.json", "r") as f:
        example_numbers = json.load(f)
        if tissue not in example_numbers:
            raise KeyError(f"Tissue '{tissue}' not found in total_dataset_sample_counts.json")
        else:
            return example_numbers[tissue]


def get_total_sample_counts(tissues: List[str]) -> int:
    """
    Get the total number of samples in the dataset.
    """
    total_count = 0
    for tissue in tissues:
        with open("/home/hauke.schuele/scGPT_distributed/total_dataset_sample_counts.json", "r") as f:
            example_numbers = json.load(f)
            if tissue not in example_numbers:
                raise KeyError(f"Tissue '{tissue}' not found in total_dataset_sample_counts.json")
            else:
                count = example_numbers[tissue]
            total_count += count
    return total_count


def get_total_batch_count(tissues: List[str], batch_size: int) -> int:
    """
    Get the total number of batches in the dataset.
    """
    total_count = get_total_sample_counts(tissues)
    return total_count // batch_size


def get_total_batch_count_per_GPU(tissues: List[str], batch_size: int) -> int:
    """
    Get the total number of batches in the dataset per GPU.
    """
    total_count = get_total_sample_counts(tissues)
    return total_count // (batch_size * WORLD_SIZE)


def create_datasets_streaming(
    data_paths: List[str], 
    validation_ratio: float,
    subset_ratio: float = 1.0,
) -> Tuple[Dataset, Dataset]:
    """
    Load the dataset from the specified data source in streaming mode.
    data_source can contain multiple directories. See argparser.py for more details
    This assumes one or more directories with a preprocessed cls_prefix_data.parquet file. 
    Returns an interleaved training dataset and a validation dataset from all directories.

    args:
        data_paths: List[Path]: 
            The paths to the data directories.
        validation_ratio: float: 
            The ratio of the dataset to use for validation.
            Uses every 1/validation_ratio shard for validation.
            Note: this is approximate, as the number of shards may not be divisible by validation_ratio.
            Note: this might result in uneven shard distribution across GPUs if n_shards in a tissue is not divisible by 1/validation_ratio.
            Best Example: For 25 shards, use validation_ratio=0.04 as 1/0.04=25 and thus every 25th shard will be used for validation.
        subset_ratio: float: (optional, default=1.0)
            The ratio of the dataset to use for training.
    returns:
        train_dataset: Dataset
            The interleaved training dataset.
        validation_dataset: Dataset
            The interleaved validation dataset.
    """
    if not data_paths:
        raise ValueError("No data_paths provided")
    if subset_ratio < 1.0:
        raise ValueError("subset_ratio < 1.0 not yet supported for streaming datasets")
        # TODO: implement subset_ratio for streaming datasets

    train_datasets = []
    validation_datasets = []

    for path in data_paths:
        files = list(path.glob("shard_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No shards found for path {path}")
        random.shuffle(files)
        validation_threshold = math.ceil(len(files) * validation_ratio) if validation_ratio > 0 else 0
        training_files = files[validation_threshold:]
        validation_files = files[:validation_threshold] if validation_ratio > 0 else []
        training_shards = [str(file) for i, file in enumerate(training_files) if (i % WORLD_SIZE) == GLOBAL_RANK]
        validation_shards = [str(file) for file in validation_files]
        printgpu(f"Found {len(files)} shards in {path} with {len(training_shards)} training shards and {len(validation_files)} validation shards assigned to this GPU (rank {GLOBAL_RANK})")
        printgpu(f"validation shards: {validation_shards}")

        if not training_shards:
            raise FileNotFoundError(f"No shards found for path {path} on rank {GLOBAL_RANK}")

        train_dataset = load_dataset(
            "parquet",
            data_files=training_shards,
            split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
            cache_dir=str(path / "cache"),
            streaming=True,  # to avoid loading all data into memory at once
        )

        if len(validation_files) > 0:
            validation_dataset = load_dataset(
                "parquet",
                data_files=validation_shards,
                split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
                cache_dir=str(path / "cache"),
                streaming=True,  # to avoid loading all data into memory at once
            )
        else:
            validation_dataset = None
            printmaster(f"No validation shards found for path {path} because validation_ratio <= 0")

        train_datasets.append(train_dataset)
        if validation_dataset:
            validation_datasets.append(validation_dataset)
        printmaster(f"tissue: {path} train_dataset.n_shards: {train_dataset.n_shards}")
        printmaster(f"tissue: {path} validation_dataset.n_shards: {validation_dataset.n_shards if validation_dataset else 0}")

    printmaster(f"Number of training datasets to concatenate: {len(train_datasets)}")
    train_dataset = concatenate_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    train_dataset = train_dataset.with_format("torch")
    printmaster(f"total train_dataset.n_shards: {train_dataset.n_shards}")


    if len(validation_datasets) > 0:
        printmaster(f"Number of validation datasets to concatenate: {len(validation_datasets)}")
        validation_dataset = concatenate_datasets(validation_datasets) if len(validation_datasets) > 1 else validation_datasets[0]
        validation_dataset = validation_dataset.with_format("torch") if validation_dataset else None
        printmaster(f"total validation_dataset.n_shards: {validation_dataset.n_shards}")
    else:
        validation_dataset = None
        printmaster(f"No validation datasets found because validation_ratio <= 0")

    return train_dataset, validation_dataset


def create_datasets(data_paths: List[Path], validation_ratio: float = 0.0, subset_ratio: float = 1.0) -> Tuple[Dataset, Dataset]:
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
    for file in data_paths:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")

        raw_dataset = load_dataset(
            "parquet",
            data_files=str(file),
            split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
            cache_dir=str(file.parent / "cache"),
        )
        if subset_ratio < 1.0:
            subset_size = int(len(raw_dataset) * subset_ratio)
            raw_dataset = raw_dataset.shuffle(seed=SEED).select(range(subset_size))

        datasets.append(raw_dataset)
        printmaster(f"Loaded {len(raw_dataset)} examples from {file}" + 
                    (f" because of subset_ratio={subset_ratio}" if subset_ratio < 1.0 else ""))
        example_count += len(raw_dataset)
    printmaster(f"Total examples across all datasets: {example_count}")
    if subset_ratio < 1.0:
        printmaster(f"Note: subset_ratio={subset_ratio} applied to each dataset")
    merged_dataset = concatenate_datasets(datasets)
    merged_dataset = merged_dataset.with_format("torch")
    train_dataset, validation_dataset = merged_dataset.train_test_split(test_size=validation_ratio, shuffle=True, seed=SEED).values()
    return train_dataset, validation_dataset


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


def create_dataloaders(train_dataset: Dataset, validation_dataset: Dataset, batch_size: int, collator: scg.DataCollator, streaming: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for training and validation datasets.
    Uses DistributedSampler for the training loader and uses the full validation dataset for the validation loader.
    """
    if streaming:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=min(CPUS_PER_TASK, train_dataset.n_shards),
            pin_memory=True,
            drop_last=True,
            collate_fn=collator,
            prefetch_factor=2
        )
        if validation_dataset:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                num_workers=min(CPUS_PER_TASK, validation_dataset.n_shards),
                pin_memory=True,
                drop_last=False,
                collate_fn=collator,
                prefetch_factor=2
            )
        else:
            validation_loader = None
    else: 
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True)
            validation_sampler = DistributedSampler(validation_dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=False)
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
            prefetch_factor=2
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            sampler=validation_sampler,
            num_workers=CPUS_PER_TASK,
            pin_memory=True,
            drop_last=False,
            collate_fn=collator,
            prefetch_factor=2
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
        subset_ratio: float = 1.0,
        streaming: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for training and validation datasets.
    """
    if not streaming:
        train_dataset, validation_dataset = create_datasets(data_paths, validation_ratio=validation_ratio, subset_ratio=subset_ratio)
        printmaster(f"Final training dataset length: {len(train_dataset)}")
        printmaster(f"Final validation dataset length: {len(validation_dataset)}")
    else:
        train_dataset, validation_dataset = create_datasets_streaming(data_paths, subset_ratio=subset_ratio, validation_ratio=validation_ratio)
        sample_count = get_total_sample_counts(TISSUES) if TISSUES is not None else None
        printmaster("Initialized streaming train datasets with approximate sample count: " + (f"{sample_count * (1-validation_ratio)}" if sample_count is not None else "unknown"))
        printmaster("Initialized streaming validation datasets with approximate sample count: " + (f"{sample_count * validation_ratio}" if sample_count is not None else "unknown"))
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
    train_loader, validation_loader = create_dataloaders(train_dataset, validation_dataset, batch_size, collator, streaming=streaming)
    dist.barrier()
    time.sleep(2)
    if not streaming:
        printgpu(f"Train loader initialized with {len(train_loader)} batches")
        printgpu(f"Validation loader initialized with {len(validation_loader)} batches")
    else:
        printmaster("Train and Validation loaders in streaming mode - exact count not available")
    return train_loader, validation_loader



# %% Model initialization

def get_model(
        embedding_size: int,
        number_of_heads: int,
        hidden_size: int,
        number_of_layers: int,
        number_of_layers_cls: int,
        dropout: float,
        pad_token: str,
        pad_value: int,
        input_emb_style: str,
        number_of_input_bins: int,
        use_generative_training: bool,
        use_fast_transformer: bool,
        vocab: GeneVocab
    ) -> DDP:
    """
    Initialize the model and wrap it in DistributedDataParallel.
    """
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        d_model=embedding_size,
        nhead=number_of_heads,
        d_hid=hidden_size,
        nlayers=number_of_layers,
        nlayers_cls=number_of_layers_cls,
        n_cls=1, # num_types if USE_CLS else 1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=False,
        use_batch_labels=False,  # TODO: try using batch labels, may help MVC
        input_emb_style=input_emb_style,
        n_input_bins=number_of_input_bins,
        use_generative_training=use_generative_training,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
    ).to(LOCAL_RANK)
    ddp_model = DDP(model, device_ids=[LOCAL_RANK])
    printgpu(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    return ddp_model


def load_model_checkpoint(model: DDP, checkpoint_path: Path) -> None:
    """
    Load the model checkpoint from the specified path.
    """
    # TODO: implement loading of model checkpoint
    pass


def get_scheduler(
        optimizer: torch.optim.Optimizer, 
        warmup_ratio_or_steps: float, 
        train_loader: DataLoader,
        epochs: int,
        scheduler_interval: int,
        scheduler_factor: float,
        streaming: bool,
        tissues: List[str] = None,
    ) -> transformers.get_scheduler:
    """
    Get the learning rate scheduler.
    """
    if warmup_ratio_or_steps > 0:
        if streaming:
            if tissues is None:
                raise ValueError("tissues must be provided for streaming datasets")
            total_num_batches = get_total_batch_count_per_GPU(tissues, train_loader.batch_size) * epochs
        else:
            total_num_batches = len(train_loader) * epochs
        warmup_steps = (
            int(total_num_batches * warmup_ratio_or_steps)
            if warmup_ratio_or_steps < 1
            else int(warmup_ratio_or_steps)
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_num_batches,
            last_epoch=-1,
        )
        printmaster(f"Using cosine scheduler with {warmup_steps} warmup steps")
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, scheduler_interval, gamma=scheduler_factor
        )
    return scheduler


# %% Training and evaluation
def pretrain_streaming(
        model: DDP,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: transformers.get_scheduler,
        num_epochs: int,
        log_interval: int,
        save_interval: int,
        device: torch.device,
        vocab: GeneVocab,
        pad_token: str,
        mask_value: float,
        fp16_enabled: bool,
        scaler: torch.cuda.amp.GradScaler,
        grad_accu_steps: int = 1,
    ) -> None:
    """
    Train the model for the specified number of epochs.
    """
    best_val_mse = float("inf")
    best_val_mre = float("inf")
    global_iter = 0
    writer = SummaryWriter(log_dir=SAVE_DIR / "tensorboard")
    total_training_time = time.time()
    delta_training_time = time.time()
    is_last_batch = False
    for epoch in range(num_epochs):
        model.train()
        printmaster(f"Starting epoch {epoch+1}/{num_epochs}")

        # train_loader = StreamingDataLoaderWrapper(train_loader)
        for i, data_dict in enumerate(train_loader):
        # for data_dict, i, is_last_batch in train_loader:

            pcpt_gene = data_dict["pcpt_gene"].to(device)
            pcpt_expr = data_dict["pcpt_expr"].to(device)
            gen_gene = data_dict["gen_gene"].to(device)
            gen_expr_target = data_dict["gen_expr_target"].to(device)
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[pad_token])
            gen_key_padding_mask = gen_gene.eq(vocab[pad_token])

            if i == 0:
                [printmaster(f"data_dict key: {k}, shape: {v.shape}") for k, v in data_dict.items()]
            
            with torch.cuda.amp.autocast(enabled=fp16_enabled):
                output_dict = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    CLS=USE_CLS,
                    MVC=MVC,
                    generative_training=True,
                )
                if i == 0:
                    [printmaster(f"output_dict key: {k}, shape: {v.shape}") for k, v in output_dict.items()]

                positions_to_match = ~gen_key_padding_mask
                loss_mse = criterion(
                    output_dict["gen_preds"], 
                    gen_expr_target, 
                    positions_to_match
                )
                loss_mvc = criterion(
                    output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
                    gen_expr_target,
                    positions_to_match,
                )

                loss_gen = torch.tensor(0.0, device=device)
                if global_iter > 1000:
                    previous_cell_embs = output_dict["cell_emb"].detach()
                    preds = model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        gen_gene,
                        gen_key_padding_mask,
                        CLS=False,
                        MVC=False,
                        input_cell_emb=previous_cell_embs,
                        generative_training=True,
                    )["gen_preds"]
                    loss_gen = criterion(preds, gen_expr_target, positions_to_match)

                total_loss = loss_mse + loss_mvc + loss_gen

            if grad_accu_steps > 1:
                total_loss = total_loss / grad_accu_steps
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            if grad_accu_steps > 1:
                if (i + 1) % grad_accu_steps == 0 or is_last_batch:
                    scheduler.step()
            else:
                scheduler.step()

            if is_master_gpu() and global_iter % log_interval == 0 and global_iter > 0:
                writer.add_scalar("loss/mse", loss_mse, global_iter)
                writer.add_scalar("loss/mvc", loss_mvc, global_iter)
                writer.add_scalar("loss/gen", loss_gen, global_iter)
                writer.add_scalar("loss/total", total_loss, global_iter)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_iter)
            if is_master_gpu() and (i + 1) % log_interval == 0 and (i + 1) > 0:
                total_time_elapsed = time.time() - total_training_time
                delta_time_elapsed = time.time() - delta_training_time
                delta_training_time = time.time()
                printlogging(
                    f"Epoch {epoch+1:2d} | Iter {i+1:5d} | "
                    f"Loss: {total_loss.item():12.4f} | "
                    f"Total Time: {str(timedelta(seconds=int(total_time_elapsed)))} | "
                    f"Delta Time: {str(timedelta(seconds=int(delta_time_elapsed)))}",
                    level="training"
                )

            if ((global_iter % save_interval == 0 and global_iter > 0) or is_last_batch) and validation_loader is not None:
                val_mse, val_mre = evaluate(
                    model=model,
                    validation_loader=validation_loader,
                    device=device,
                    vocab=vocab,
                    pad_token=pad_token,
                    fp16_enabled=fp16_enabled,
                    mask_value=mask_value,
                    criterion=criterion,
                )
                writer.add_scalar("validation/mse", val_mse, global_iter)
                writer.add_scalar("validation/mre", val_mre, global_iter)
                saved = False
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    if is_master_gpu():
                        torch.save(
                            model.state_dict(),
                            SAVE_DIR / "best_model_mse.pt",
                        )
                        saved = True
                printlogging(
                    f"Epoch {epoch+1:2d} | Iter {i+1:5d} | "
                    f"Loss: {val_mse:12.4f} | "
                    f"Saved: {saved}",
                    level="validation"
                )
            global_iter += 1

    writer.close()
    printmaster("Training complete.")


def pretrain(
        model: DDP,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: transformers.get_scheduler,
        num_epochs: int,
        log_interval: int,
        save_interval: int,
        device: torch.device,
        vocab: GeneVocab,
        pad_token: str,
        mask_value: float,
        fp16_enabled: bool,
        scaler: torch.cuda.amp.GradScaler,
        grad_accu_steps: int = 1,
    ) -> None:
    """
    Train the model for the specified number of epochs.
    """
    best_val_mse = float("inf")
    best_val_mre = float("inf")
    writer = SummaryWriter(log_dir=SAVE_DIR / "tensorboard")
    total_training_time = time.time()
    delta_training_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        printmaster(f"Starting epoch {epoch+1}/{num_epochs}")
        for i, data_dict in enumerate(train_loader):
            global_iter = epoch * len(train_loader) + i

            pcpt_gene = data_dict["pcpt_gene"].to(device)
            pcpt_expr = data_dict["pcpt_expr"].to(device)
            gen_gene = data_dict["gen_gene"].to(device)
            gen_expr_target = data_dict["gen_expr_target"].to(device)
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[pad_token])
            gen_key_padding_mask = gen_gene.eq(vocab[pad_token])

            if i == 0:
                [printmaster(f"data_dict key: {k}, shape: {v.shape}") for k, v in data_dict.items()]

            with torch.cuda.amp.autocast(enabled=fp16_enabled):
                output_dict = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    CLS=USE_CLS,
                    MVC=MVC,
                    generative_training=True,
                )
                if i == 0:
                    [printmaster(f"output_dict key: {k}, shape: {v.shape}") for k, v in output_dict.items()]

                positions_to_match = ~gen_key_padding_mask
                loss_mse = criterion(
                    output_dict["gen_preds"], 
                    gen_expr_target, 
                    positions_to_match
                )
                loss_mvc = criterion(
                    output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
                    gen_expr_target,
                    positions_to_match,
                )

                loss_gen = torch.tensor(0.0, device=device)
                if global_iter > 1000:
                    previous_cell_embs = output_dict["cell_emb"].detach()
                    preds = model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        gen_gene,
                        gen_key_padding_mask,
                        CLS=False,
                        MVC=False,
                        input_cell_emb=previous_cell_embs,
                        generative_training=True,
                    )["gen_preds"]
                    loss_gen = criterion(preds, gen_expr_target, positions_to_match)

                total_loss = loss_mse + loss_mvc + loss_gen

            if grad_accu_steps > 1:
                total_loss = total_loss / grad_accu_steps
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            if grad_accu_steps > 1:
                if (i + 1) % grad_accu_steps == 0 or (i + 1) == len(train_loader):
                    scheduler.step()
            else:
                scheduler.step()

            if is_master_gpu() and global_iter % log_interval == 0 and global_iter > 0:
                writer.add_scalar("loss/mse", loss_mse, global_iter)
                writer.add_scalar("loss/mvc", loss_mvc, global_iter)
                writer.add_scalar("loss/gen", loss_gen, global_iter)
                writer.add_scalar("loss/total", total_loss, global_iter)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_iter)
            if is_master_gpu() and (i + 1) % log_interval == 0 and (i + 1) > 0:
                total_time_elapsed = time.time() - total_training_time
                delta_time_elapsed = time.time() - delta_training_time
                delta_training_time = time.time()
                printlogging(
                    f"Epoch {epoch+1:2d} | Iter {i+1:5d}/{len(train_loader)} | "
                    f"Loss: {total_loss.item():12.4f} | "
                    f"Total Time: {str(timedelta(seconds=int(total_time_elapsed)))} | "
                    f"Delta Time: {str(timedelta(seconds=int(delta_time_elapsed)))}",
                    level="training"
                )

            if (global_iter % save_interval == 0 and global_iter > 0) or (i+1) == len(train_loader):
                val_mse, val_mre = evaluate(
                    model=model,
                    validation_loader=validation_loader,
                    device=device,
                    vocab=vocab,
                    pad_token=pad_token,
                    fp16_enabled=fp16_enabled,
                    mask_value=mask_value,
                    criterion=criterion,
                )
                writer.add_scalar("validation/mse", val_mse, global_iter)
                writer.add_scalar("validation/mre", val_mre, global_iter)
                saved = False
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    if is_master_gpu():
                        torch.save(
                            model.state_dict(),
                            SAVE_DIR / "best_model_mse.pt",
                        )
                        saved = True
                printlogging(
                    f"Epoch {epoch+1:2d} | Iter {i+1:5d}/{len(train_loader)} | "
                    f"Loss: {val_mse:12.4f} | "
                    f"Saved: {saved}",
                    level="validation"
                )

    writer.close()
    printmaster("Training complete.")


def evaluate(
        model: DDP, 
        validation_loader: DataLoader, 
        device: torch.device, 
        vocab: GeneVocab,
        pad_token: str,
        fp16_enabled: bool,
        mask_value: float,
        criterion: nn.Module,
    ) -> Dict[str, torch.Tensor]:
    """
    Evaluate the model on the evaluation data.
    Expects the Validation_loader to be distributed.
    Averages the loss values across all GPUs.
    """
    model.eval()
    total_mse = 0.0
    total_mre = 0.0
    total_count = 0
    with torch.no_grad():
        for data_dict in validation_loader:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            batch_size = data_dict["pcpt_gene"].shape[0]

            pcpt_gene = data_dict["pcpt_gene"]
            pcpt_expr = data_dict["pcpt_expr"]
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[pad_token])
            gen_gene = data_dict["gen_gene"]
            gen_expr_target = target_values = data_dict["gen_expr_target"]
            gen_key_padding_mask = gen_gene.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=fp16_enabled):
                output_dict = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    CLS=False,
                    MVC=False,
                    generative_training=True,
                )
                output_values = output_dict["gen_preds"]

                positions_to_match = ~gen_key_padding_mask

                loss = criterion(output_values, target_values, positions_to_match)
                mre = masked_relative_error(
                    output_values, target_values, positions_to_match
                )

            total_mse += loss.item() * batch_size
            total_mre += mre.item() * batch_size
            total_count += batch_size
            
    if dist.is_initialized():
        dist.barrier()
        total_mse = torch.tensor(total_mse, device=device, dtype=torch.float)
        total_mre = torch.tensor(total_mre, device=device, dtype=torch.float)
        total_count = torch.tensor(total_count, device=device, dtype=torch.float)
        dist.all_reduce(total_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_mre, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        global_mse = total_mse.item() / total_count.item()
        global_mre = total_mre.item() / total_count.item()
    else:
        raise ValueError("Distributed not initialized")
    
    return global_mse, global_mre


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
    global USE_CLS, USE_CCE, MVC
    global TISSUES
    SAVE_DIR = Path(args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)
    USE_GENERATIVE_TRAINING = True if args.training_tasks in ["gen", "both"] else False
    SPECIAL_TOKENS = [args.pad_token, "<cls>", "<eoc>"]
    USE_CLS = not args.no_cls
    USE_CCE = not args.no_cce
    MVC = True
    TISSUES = args.tissues if args.streaming else None

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
        args.n_input_bins = args.n_bins + 2
    else:
        args.mask_value = -1
        args.pad_value = -2
        args.n_input_bins = args.n_bins

    if args.training_tasks in ["gen", "both"]:
        printmaster(f"args.mask_ratio: {args.mask_ratio} (can be float or list of floats)")
        # args.mask_ratio = [0.25, 0.50, 0.75]
    return args


def get_arguments():
    """
    Get command line arguments.  
    Arguments are specified in the argparser.py file.
    """
    parser = argparser.get_parser()
    args = parser.parse_args()
    argparser.validate_args(args)
    return args


# %% main


def main():
    global runtime_startTime
    runtime_startTime = time.time()
    initialize_slurm_variables()
    args = get_arguments()
    args = initialize_additional_arguments(args)
    initialize_utility_variables(args)
    dump_args(args)
    setup_distributeddataparallel()
    printmaster(f"Setup complete. Group initialized? {dist.is_initialized()}")
    try:
        total_start_time = time.time()
        # Load data
        vocab = get_vocabulary(Path(args.vocab_path))
        dump_vocab(vocab)
        train_loader, validation_loader = get_dataloaders(
            data_paths=argparser.get_datapaths(args), 
            validation_ratio=args.valid_ratio, 
            vocab=vocab,
            max_seq_len=args.max_seq_len,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            input_style=args.input_style,
            mask_ratio=args.mask_ratio,
            mask_value=args.mask_value,
            trunc_by_sample=args.trunc_by_sample,
            training_tasks=args.training_tasks,
            batch_size=args.batch_size,
            subset_ratio=args.subset_ratio,
            streaming=args.streaming,
        )
        dist.barrier()
        time.sleep(2)
        # Initialize model, criterion, optimizer, scheduler
        ddp_model = get_model(
            embedding_size=args.embsize,
            number_of_heads=args.nheads,
            hidden_size=args.d_hid,
            number_of_layers=args.nlayers,
            number_of_layers_cls=args.n_layers_cls,
            dropout=args.dropout,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            input_emb_style=args.input_emb_style,
            number_of_input_bins=args.n_input_bins,
            use_generative_training=True if args.training_tasks in ["gen", "both"] else False,
            use_fast_transformer=args.fast_transformer,
            vocab=vocab
        )
        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
        scheduler = get_scheduler(
            optimizer=optimizer,
            warmup_ratio_or_steps=args.warmup_ratio_or_steps,
            train_loader=train_loader,
            epochs=args.epochs,
            scheduler_interval=args.scheduler_interval,
            scheduler_factor=args.scheduler_factor,
            streaming=args.streaming,
            tissues=args.tissues if args.streaming else None,
        )
        dist.barrier()
        time.sleep(2)
        # start training
        if args.streaming:
            pretrain_streaming(
                model=ddp_model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=args.epochs,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                device=torch.device(LOCAL_RANK),
                vocab=vocab,
                pad_token=args.pad_token,
                mask_value=args.mask_value,
                fp16_enabled=args.fp16,
                scaler=torch.cuda.amp.GradScaler(enabled=args.fp16),
                grad_accu_steps=args.grad_accu_steps,
            )
        else:
            pretrain(
                model=ddp_model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=args.epochs,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                device=torch.device(LOCAL_RANK),
                vocab=vocab,
                pad_token=args.pad_token,
                mask_value=args.mask_value,
                fp16_enabled=args.fp16,
                scaler=torch.cuda.amp.GradScaler(enabled=args.fp16),
                grad_accu_steps=args.grad_accu_steps,
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        total_elapsed = time.time() - total_start_time
        printgpu(f"Exiting after {str(timedelta(seconds=total_elapsed))} hours.")


if __name__ == "__main__":
    main()