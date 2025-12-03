

### Imports

# systen relevant imports
import math
import os
import sys
import argparse
import json
import time
import random
import re
from socket import gethostname
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
import pyarrow.parquet as pq

# torch relevant imports
import torch
import transformers
from torch import mode, nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import Dataset, load_dataset, concatenate_datasets

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# scgpt imports
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.tokenizer import GeneVocab, random_mask_value
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
    if SEPARATE_LOG_FILES:
        with open(SAVE_DIR / f"GPU[{GLOBAL_RANK}]_log.txt", "a") as f:
            current_runtime = time.time() - runtime_startTime
            current_runtime = str(timedelta(seconds=int(current_runtime)))
            f.write(f"[{current_runtime}]|[GPU:{GLOBAL_RANK}][{NODE_ID}:{LOCAL_RANK}] {string}\n")
    else:
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


def get_total_sample_counts(data_paths: List[str]) -> int:
    """
    Get the total number of samples in the dataset.
    """
    total_count = 0
    for path in data_paths:
        total_count += pq.ParquetFile(path).metadata.num_rows
    return total_count


def get_total_batch_count(data_paths: List[str], batch_size: int) -> int:
    """
    Get the total number of batches in the dataset.
    """
    total_count = get_total_sample_counts(data_paths)
    return total_count // batch_size


def get_total_batch_count_per_GPU(data_paths: List[str], batch_size: int) -> int:
    """
    Get the total number of batches in the dataset per GPU.
    """
    total_count = get_total_sample_counts(data_paths)
    return (total_count // (batch_size * WORLD_SIZE)) - 1 # To account for drop_last=True, don't @ me, I know that // already accounts for that


def validate_data_paths(data_paths: List[Path]) -> None:
    """
    Validate that the data paths exist.
    """
    for path in data_paths:
        if not path.exists():
            raise FileNotFoundError(f"Data path '{path}' does not exist.")
        if not path.is_file():
            raise ValueError(f"Data path '{path}' is not a file.")
    if len(data_paths) % WORLD_SIZE != 0:
        raise ValueError(f"Number of data paths ({len(data_paths)}) is not divisible by WORLD_SIZE ({WORLD_SIZE}).")
    num_rows = pq.ParquetFile(data_paths[0]).metadata.num_rows
    for path in data_paths[1:]:
        if pq.ParquetFile(path).metadata.num_rows != num_rows:
            raise ValueError(f"Data path '{path}' has a different number of rows ({pq.ParquetFile(path).metadata.num_rows}) than the first data path ({num_rows}).")


def validate_shard_distribution(shards: List[str], shard_type: str = "") -> None:
    """
    Validate that the shards are evenly distributed across GPUs.
    """
    if not dist.is_initialized():
        raise ValueError("Distributed not initialized")
    count = torch.tensor(len(shards), device=LOCAL_RANK, dtype=torch.long)

    gathered_shards = [torch.zeros_like(count) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_shards, count)
    if not all(count.item() == gathered_shards[0].item() for count in gathered_shards):
        raise ValueError(f"{shard_type} Shards are not evenly distributed across GPUs: {gathered_shards}")
    
    
def validate_sample_distribution(shards: List[str], sample_type: str = "") -> None:
    """
    Validate that the samples are evenly distributed across GPUs.
    """
    if not dist.is_initialized():
        raise ValueError("Distributed not initialized")
    total_size = torch.tensor(sum(pq.ParquetFile(path).metadata.num_rows for path in shards), device=LOCAL_RANK, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(total_size) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_sizes, total_size)
    if not all(size.item() == gathered_sizes[0].item() for size in gathered_sizes):
        raise ValueError(f"{sample_type} Samples are not evenly distributed across GPUs: {gathered_sizes}")



# def create_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset]:
#     """
#     Load the dataset from the specified data source in streaming mode.
#     data_source is a list of .parquet files. 
#     Each file is assumed to be a shard of the dataset and should be of equal size.

#     args:
#         data_paths: List[Path]: 
#             The paths to the data directories.
#         validation_ratio: float: 
#             The ratio of the dataset to use for validation.
#             Uses every 1/validation_ratio shard for validation.
#             Note: this is approximate, as the number of shards may not be divisible by validation_ratio.
#             Note: this might result in uneven shard distribution across GPUs if n_shards in a tissue is not divisible by 1/validation_ratio.
#             Best Example: For 25 shards, use validation_ratio=0.04 as 1/0.04=25 and thus every 25th shard will be used for validation.
#         subset_ratio: float: (optional, default=1.0)
#             The ratio of the dataset to use for training.
#     returns:
#         train_dataset: Dataset
#             The interleaved training dataset.
#         validation_dataset: Dataset
#             The interleaved validation dataset.
#     """
#     if not args.data_paths:
#         raise ValueError("No data_paths provided")
#     if args.subset_ratio < 1.0:
#         raise ValueError("subset_ratio < 1.0 not yet supported for streaming datasets")
#         # TODO: implement subset_ratio for streaming datasets

#     training_files = args.train_paths
#     validation_files = args.valid_paths
#     printmaster(f"Total number of shards: {len(args.data_paths)}")
#     printmaster(f"Number of training shards: {len(training_files)}")
#     printmaster(f"Number of validation shards: {len(validation_files)}")
#     training_shards = [str(file) for i, file in enumerate(training_files) if (i % WORLD_SIZE) == GLOBAL_RANK]
#     if len(validation_files) % WORLD_SIZE == 0:
#         validation_shards = [str(file) for i, file in enumerate(validation_files) if (i % WORLD_SIZE) == GLOBAL_RANK]
#     else:
#         printmaster(f"Warning: number of validation shards {len(validation_files)} is not divisible by WORLD_SIZE {WORLD_SIZE}, so every GPU is getting all validation shards, which is not ideal.")
#         validation_shards = [str(file) for file in validation_files]
#     printgpu(f"training_shards ({len(training_shards)}): {json.dumps(training_shards, indent=2)}")
#     printgpu(f"validation_shards ({len(validation_shards)}): {json.dumps(validation_shards, indent=2)}")

#     if not training_shards:
#         raise FileNotFoundError(f"No shards found on rank {GLOBAL_RANK}")

#     validate_shard_distribution(training_shards, "Training")
#     validate_sample_distribution(training_shards, "Training")
#     validate_shard_distribution(validation_shards, "Validation")
#     validate_sample_distribution(validation_shards, "Validation")

#     train_dataset = load_dataset(
#         "parquet",
#         data_files=training_shards,
#         split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
#         streaming=True,
#     )
#     if args.shuffle_buffer_size > 0:
#         train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=42)
#     train_dataset = train_dataset.with_format("torch")

#     if len(validation_files) > 0:
#         validation_dataset = load_dataset(
#             "parquet",
#             data_files=validation_shards,
#             split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
#             streaming=True,
#         )
#         validation_dataset = validation_dataset.with_format("torch")
#     else:
#         validation_dataset = None
#         printmaster(f"No validation shards found because validation_ratio <= 0")

#     return train_dataset, validation_dataset


def create_train_dataset(args: argparse.Namespace, epoch: int) -> Dataset:
    if not args.data_paths:
        raise ValueError("No data_paths provided")

    training_files = args.train_paths
    random.Random(SEED + epoch).shuffle(training_files)
    printmaster(f"Total number of shards: {len(args.data_paths)}")
    printmaster(f"Number of training shards: {len(training_files)}")
    training_shards = [str(file) for i, file in enumerate(training_files) if (i % WORLD_SIZE) == GLOBAL_RANK]
    printgpu(f"training_shards ({len(training_shards)}): {json.dumps(training_shards, indent=2)}")

    if not training_shards:
        raise FileNotFoundError(f"No shards found on rank {GLOBAL_RANK}")

    validate_shard_distribution(training_shards, "Training")
    validate_sample_distribution(training_shards, "Training")

    train_dataset = load_dataset(
        "parquet",
        data_files=training_shards,
        split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
        streaming=True,
    )
    if args.shuffle_buffer_size > 0:
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=42)
    train_dataset = train_dataset.with_format("torch")
    return train_dataset


def create_validation_dataset(args: argparse.Namespace) -> Dataset:
    validation_files = args.valid_paths
    printmaster(f"Number of validation shards: {len(validation_files)}")
    if len(validation_files) % WORLD_SIZE == 0:
        validation_shards = [str(file) for i, file in enumerate(validation_files) if (i % WORLD_SIZE) == GLOBAL_RANK]
    else:
        printmaster(f"Warning: number of validation shards {len(validation_files)} is not divisible by WORLD_SIZE {WORLD_SIZE}, so every GPU is getting all validation shards, which is not ideal.")
        validation_shards = [str(file) for file in validation_files]
    printgpu(f"validation_shards ({len(validation_shards)}): {json.dumps(validation_shards, indent=2)}")
    validate_shard_distribution(validation_shards, "Validation")
    validate_sample_distribution(validation_shards, "Validation")
    validation_dataset = load_dataset(
        "parquet",
        data_files=validation_shards,
        split="train",  # specify train to load all the data into a dataset directly and not a dataset dict
        streaming=True,
    )
    validation_dataset = validation_dataset.with_format("torch")
    return validation_dataset


def create_dataloader(dataset: Dataset, collator: scg.DataCollator, batch_size: int) -> DataLoader:    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(CPUS_PER_TASK, dataset.n_shards),
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        prefetch_factor=2
    )
    return dataloader


def get_train_loader(args: argparse.Namespace, vocab: GeneVocab, epoch: int) -> DataLoader:
    dataset = create_train_dataset(args, epoch)
    collator = create_collator(args=args, vocab=vocab)
    dataloader = create_dataloader(dataset, collator, args.batch_size)
    dist.barrier()
    time.sleep(2)
    return dataloader


def get_validation_loader(args: argparse.Namespace, vocab: GeneVocab) -> DataLoader:
    dataset = create_validation_dataset(args)
    collator = create_collator(args=args, vocab=vocab)
    dataloader = create_dataloader(dataset, collator, args.batch_size)
    printmaster(f"Validation loader prefetch factor: {dataloader.prefetch_factor}, num_workers: {dataloader.num_workers}")
    dist.barrier()
    time.sleep(2)
    return dataloader


def create_collator(
        args: argparse.Namespace,
        vocab: GeneVocab
    ) -> scg.DataCollator:
    """
    Create a data collator for the dataloader with provided arguments.
    """
    collator = scg.DataCollator(
        do_padding=True if args.max_seq_len is not None else False,
        pad_token_id=vocab[args.pad_token],
        pad_value=args.pad_value,
        do_mlm=True,
        do_binning=True if args.input_style == "binned" else False,
        mlm_probability=args.mask_ratio,
        mask_value=args.mask_value,
        max_length=args.max_seq_len,
        sampling=args.trunc_by_sample,
        data_style=args.training_tasks,
    )
    return collator


# def create_dataloaders(train_dataset: Dataset, validation_dataset: Dataset, batch_size: int, collator: scg.DataCollator) -> Tuple[DataLoader, DataLoader]:
#     """
#     Get the dataloaders for training and validation datasets.
#     Uses DistributedSampler for the training loader and uses the full validation dataset for the validation loader.
#     """
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         num_workers=min(CPUS_PER_TASK, train_dataset.n_shards),
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=collator,
#         prefetch_factor=2
#     )
#     if validation_dataset:
#         validation_loader = DataLoader(
#             validation_dataset,
#             batch_size=batch_size,
#             num_workers=min(CPUS_PER_TASK, validation_dataset.n_shards),
#             pin_memory=True,
#             drop_last=False,
#             collate_fn=collator,
#             prefetch_factor=2
#         )
#         printmaster(f"Validation loader prefetch factor: {validation_loader.prefetch_factor}, num_workers: {validation_loader.num_workers}")
#     else:
#         validation_loader = None
#     return train_loader, validation_loader


# def get_dataloaders(args: argparse.Namespace, vocab: GeneVocab) -> Tuple[DataLoader, DataLoader]:
#     """
#     Get the dataloaders for training and validation datasets.
#     """
#     train_dataset, validation_dataset = create_datasets(args=args)
#     collator = create_collator(
#         vocab=vocab,
#         max_seq_len=args.max_seq_len,
#         pad_token=args.pad_token,
#         pad_value=args.pad_value,
#         input_style=args.input_style,
#         mask_ratio=args.mask_ratio,
#         mask_value=args.mask_value,
#         trunc_by_sample=args.trunc_by_sample,
#         training_tasks=args.training_tasks,
#     )
#     train_loader, validation_loader = create_dataloaders(train_dataset, validation_dataset, args.batch_size, collator)
#     dist.barrier()
#     time.sleep(2)
#     return train_loader, validation_loader



# %% Model initialization

def get_model(args: argparse.Namespace, vocab: GeneVocab) -> DDP:
    """
    Initialize the model and wrap it in DistributedDataParallel.
    """
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        d_model=args.embsize,
        nhead=args.nheads,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        nlayers_cls=args.n_layers_cls,
        n_cls=1, # num_types if USE_CLS else 1,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=args.pad_token,
        pad_value=args.pad_value,
        do_mvc=MVC,
        do_dab=False,
        use_batch_labels=False,  # TODO: try using batch labels, may help MVC
        input_emb_style=args.input_emb_style,
        n_input_bins=args.n_input_bins,
        use_generative_training=USE_GENERATIVE_TRAINING,
        use_fast_transformer=args.fast_transformer,
        fast_transformer_backend="flash",
    ).to(LOCAL_RANK)
    ddp_model = DDP(model, device_ids=[LOCAL_RANK])
    printgpu(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):_} trainable parameters")
    return ddp_model


def get_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> transformers.get_scheduler:
    """
    Get the learning rate scheduler.
    """
    if args.warmup_ratio_or_steps > 0:
        if args.train_paths is None:
            raise ValueError("data_paths must be provided for streaming datasets")
        total_num_batches = get_total_batch_count_per_GPU(args.train_paths, args.batch_size) * args.epochs
        warmup_steps = (
            int(total_num_batches * args.warmup_ratio_or_steps)
            if args.warmup_ratio_or_steps < 1
            else int(args.warmup_ratio_or_steps)
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
            optimizer, args.scheduler_interval, gamma=args.scheduler_factor
        )
    return scheduler


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Get the latest training state file.
    """
    state_files = list(Path(checkpoint_dir).glob("checkpoint-*.pt"))
    if not state_files:
        raise FileNotFoundError(f"No training state files found in {checkpoint_dir}.")
    regex = r'checkpoint-(?P<epoch>\d+)-(?P<batch_idx>\d+)\.pt'
    latest_epoch = -1
    latest_batch_idx = -1
    latest_file = None
    for file in state_files:
        m = re.search(regex, file.name)
        if m:
            epoch = int(m.group('epoch'))
            batch_idx = int(m.group('batch_idx'))
            if (epoch > latest_epoch) or (epoch == latest_epoch and batch_idx > latest_batch_idx):
                latest_epoch = epoch
                latest_batch_idx = batch_idx
                latest_file = file
    return str(latest_file)


def load_checkpoint(
        checkpoint_dir: str,
        model: DDP,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: transformers.get_scheduler,
) -> None:
    """
    Load the training state from the specified directory.
    """
    state_path = get_latest_checkpoint(checkpoint_dir)
    if not Path(state_path).is_file():
        raise FileNotFoundError(f"Could not find {state_path} in {checkpoint_dir}.")
    printmaster(f"Loading training state from {state_path}...")
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict["model_state"])
    random.setstate(state_dict["py_random_state"])
    torch.set_rng_state(state_dict["torch_random_state"].to("cpu"))
    torch.cuda.set_rng_state_all([cuda_random_state.to("cpu") for cuda_random_state in state_dict["cuda_random_state"]])
    optimizer.load_state_dict(state_dict["optimizer_state"])
    scaler.load_state_dict(state_dict["scaler_state"])
    scheduler.load_state_dict(state_dict["scheduler_state"])
    printmaster(f"Finished loading training state from {state_path}")
    return state_dict["global_iter"], state_dict["epoch"], state_dict["batch_idx"]


# %% Training and evaluation
def pretrain(
        args: argparse.Namespace,
        model: DDP,
        # train_loader: DataLoader,
        # validation_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: transformers.get_scheduler,
        device: torch.device,
        vocab: GeneVocab,
        scaler: torch.cuda.amp.GradScaler,
    ) -> None:
    """
    Train the model for the specified number of epochs.
    """
    best_val_mse = float("inf")
    best_val_mre = float("inf")
    writer = SummaryWriter(log_dir=SAVE_DIR / "tensorboard")
    training_start_time = time.time()
    delta_training_time = time.time()

    is_last_batch = False
    n_total_batches=get_total_batch_count_per_GPU(args.train_paths, args.batch_size)
    global_iter, epoch_offset, batch_offset = 0, 0, 0
    if args.checkpoint_dir is not None:
        global_iter, epoch_offset, batch_offset = load_checkpoint(args.checkpoint_dir, model, device, optimizer, scaler, scheduler)
    validation_loader = get_validation_loader(args, vocab)
    for epoch in range(args.epochs):
        model.train()
        if epoch < epoch_offset:
            printmaster(f"Skipping epoch {epoch+1} due to continue training from epoch {epoch_offset}")
            continue
        printmaster(f"Starting epoch {epoch+1}/{args.epochs}")
        train_loader = get_train_loader(args, vocab, epoch)

        if args.shuffle_buffer_size > 0:
            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            elif i == 0:
                printmaster("Warning: train_loader.dataset has no set_epoch method, shuffling may not be deterministic across epochs.")
        running_loss_mse = 0.0
        running_loss_mvc = 0.0
        running_loss_gen = 0.0
        running_loss_total = 0.0

        for i, data_dict in enumerate(train_loader):
            if i < batch_offset and epoch == epoch_offset:
                if is_master_gpu() and (i + 1) % args.log_interval == 0:
                    printlogging(f"Skipping batch {i+1} due to continue training from batch {batch_offset}", level="Training")
                continue

            pcpt_gene = data_dict["pcpt_gene"].to(device)
            pcpt_expr = data_dict["pcpt_expr"].to(device)
            gen_gene = data_dict["gen_gene"].to(device)
            gen_expr_target = data_dict["gen_expr_target"].to(device)
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
            gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])

            if i == 0:
                [printmaster(f"data_dict key: {k}, shape: {v.shape}") for k, v in data_dict.items()]
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
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

            if args.grad_accu_steps > 1:
                total_loss = total_loss / args.grad_accu_steps
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )  # gradient clipping
            if not math.isfinite(total_loss):
                print(f"Non-finite loss detected: {total_loss.item()} at epoch {epoch}, batch {i}. Skipping batch and continuing training.")
                continue
            scaler.step(optimizer)
            scaler.update()
            if args.grad_accu_steps > 1:
                if (i + 1) % args.grad_accu_steps == 0 or is_last_batch:
                    scheduler.step()
            else:
                scheduler.step()


            running_loss_mse += loss_mse.item()
            running_loss_mvc += loss_mvc.item()
            running_loss_gen += loss_gen.item()
            running_loss_total += total_loss.item()
            if ((i+1) % args.log_interval == 0):
                delta_training_time = log_training(args, scheduler, writer, training_start_time, delta_training_time, n_total_batches, global_iter, epoch, running_loss_mse, running_loss_mvc, running_loss_gen, running_loss_total, i)
                running_loss_mse, running_loss_mvc, running_loss_gen, running_loss_total = 0.0, 0.0, 0.0, 0.0

            if ((i+1) % args.save_interval == 0) and validation_loader is not None:
                best_val_mse, delta_training_time = eval_and_save(args, model, train_loader, validation_loader, criterion, optimizer, scheduler, device, vocab, scaler, best_val_mse, writer, training_start_time, delta_training_time, global_iter, epoch, i)
            global_iter += 1
            dist.barrier()

        delta_training_time = log_training(args, scheduler, writer, training_start_time, delta_training_time, n_total_batches, global_iter, epoch, running_loss_mse, running_loss_mvc, running_loss_gen, running_loss_total, i)
        running_loss_mse, running_loss_mvc, running_loss_gen, running_loss_total = 0.0, 0.0, 0.0, 0.0
        if validation_loader is not None:
            best_val_mse, delta_training_time = eval_and_save(args, model, train_loader, validation_loader, criterion, optimizer, scheduler, device, vocab, scaler, best_val_mse, writer, training_start_time, delta_training_time, global_iter, epoch+1, 0)
        dist.barrier()
    writer.close()
    printmaster("Training complete.")


def log_training(args, scheduler, writer, training_start_time, delta_training_time, n_total_batches, global_iter, epoch, running_loss_mse, running_loss_mvc, running_loss_gen, running_loss_total, i):
    if is_master_gpu():
        total_time_elapsed = time.time() - training_start_time
        delta_time_elapsed = time.time() - delta_training_time
        delta_training_time = time.time()
        printlogging(
            f"Epoch {epoch+1:2d}/{args.epochs:2d} | Iter {i+1:5d}/{n_total_batches:5d} | "
            f"Loss: {running_loss_total / args.log_interval:12.4f} | "
            f"Total Time: {str(timedelta(seconds=int(total_time_elapsed)))} | "
            f"Delta Time: {str(timedelta(seconds=int(delta_time_elapsed)))}",
            level="training"
        )
        writer.add_scalar("loss/mse", running_loss_mse / args.log_interval, global_iter)
        writer.add_scalar("loss/mvc", running_loss_mvc / args.log_interval, global_iter)
        writer.add_scalar("loss/gen", running_loss_gen / args.log_interval, global_iter)
        writer.add_scalar("loss/total", running_loss_total / args.log_interval, global_iter)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_iter)
    # if SEPARATE_LOG_FILES:
    #     total_time_elapsed = time.time() - total_training_time
    #     delta_time_elapsed = time.time() - delta_training_time
    #     delta_training_time = time.time()
    #     printgpu(
    #         f"Epoch {epoch+1:2d}/{args.epochs:2d} | Iter {i+1:5d}/{n_total_batches:5d} | "
    #         f"Loss: {running_loss_total / args.log_interval:12.4f} | "
    #         f"Total Time: {str(timedelta(seconds=int(total_time_elapsed)))} | "
    #         f"Delta Time: {str(timedelta(seconds=int(delta_time_elapsed)))}"
    #     )
        return delta_training_time
    return 0


def eval_and_save(args, model, train_loader, validation_loader, criterion, optimizer, scheduler, device, vocab, scaler, best_val_mse, writer, training_start_time, delta_training_time, global_iter, epoch, i):
    val_mse, val_mre = evaluate(
        model=model,
        validation_loader=validation_loader,
        device=device,
        vocab=vocab,
        pad_token=args.pad_token,
        fp16_enabled=args.fp16,
        mask_value=args.mask_value,
        criterion=criterion,
    )
    if is_master_gpu():
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
                
        total_time_elapsed = time.time() - training_start_time
        delta_time_elapsed = time.time() - delta_training_time
        delta_training_time = time.time()
        printlogging(
            f"Epoch {epoch+1:2d} | Iter {i+1:5d} | "
            f"MSE: {val_mse:4.4f} | "
            f"MRE: {val_mre:4.4f} | "
            f"Saved: {saved} | "
            f"Total Time: {str(timedelta(seconds=int(total_time_elapsed)))} | "
            f"Delta Time: {str(timedelta(seconds=int(delta_time_elapsed)))}",
            level="validation"
    )
    # if SEPARATE_LOG_FILES:
    #     printgpu(
    #         f"Epoch {epoch+1:2d}/{args.epochs:2d} | Iter {i+1:5d} | "
    #         f"Loss: {val_mse:12.4f} | "
    #         f"Saved: {saved}"
    #     )
    # if is_master_gpu():
        state_dict = {
            "epoch": epoch,
            "batch_idx": i,
            "global_iter": global_iter,
            "model_state": model.state_dict(),
            "dataloader_state": train_loader.state_dict() if hasattr(train_loader, "state_dict") else None,
            "py_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state().cpu(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(state_dict, SAVE_DIR / f"checkpoint-{epoch+1}-{i+1}.pt")
    return best_val_mse, delta_training_time


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
    # add UMAP after evaluation and metrics from scpgt evaluation (integration tutorial)
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
    torch.cuda.set_device(LOCAL_RANK)


def initialize_utility_variables(args: argparse.Namespace):
    """
    Initialize utility variables from the arguments globally.
    """
    global SAVE_DIR, USE_GENERATIVE_TRAINING, SPECIAL_TOKENS, SEED
    global USE_CLS, USE_CCE, MVC
    global TISSUES
    global SEPARATE_LOG_FILES
    SEPARATE_LOG_FILES = args.separate_gpu_log_files
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


def get_split_paths(args: argparse.Namespace) -> Tuple[List[Path], List[Path]]:
    """
    Get the training and validation data paths from the command line arguments.
    """
    random.shuffle(args.data_paths)
    validation_threshold = math.ceil(len(args.data_paths) * args.valid_ratio) if args.valid_ratio > 0 else 0
    training_files = args.data_paths[validation_threshold:]
    validation_files = args.data_paths[:validation_threshold] if args.valid_ratio > 0 else []
    return training_files, validation_files


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
        args.mask_ratio = [0.25, 0.50, 0.75]
        printmaster(f"args.mask_ratio: {args.mask_ratio} (can be float or list of floats)")
    args.data_paths = argparser.get_datapaths(args)
    args.train_paths, args.valid_paths = get_split_paths(args)
    return args


def get_arguments():
    """
    Get command line arguments.  
    Arguments are specified in the argparser.py file.
    """
    args = argparser._parse_args()
    argparser.validate_args(args)
    return args


# %% main


def main():
    global runtime_startTime
    runtime_startTime = time.time()
    initialize_slurm_variables()
    args = get_arguments()
    if not args.checkpoint_dir:
        args = initialize_additional_arguments(args)
    else:
        printmaster(f"----------------------------------------------------------------------------")
        printmaster(f"Continuing training from directory: {args.checkpoint_dir}!!!")
        printmaster(f"----------------------------------------------------------------------------")
    initialize_utility_variables(args)
    dump_args(args)
    setup_distributeddataparallel()
    printmaster(f"Setup complete. Group initialized? {dist.is_initialized()}")
    try:
        total_start_time = time.time()
        # Load data
        vocab = get_vocabulary(Path(args.vocab_path))
        dump_vocab(vocab)
        # train_loader, validation_loader = get_dataloaders(args=args, vocab=vocab)
        dist.barrier()
        time.sleep(2)
        # Initialize model, criterion, optimizer, scheduler
        ddp_model = get_model(args=args, vocab=vocab)
        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
        scheduler = get_scheduler(
            args=args,
            optimizer=optimizer,
        )
        dist.barrier()
        time.sleep(2)
        # start training
        pretrain(
            args=args,
            model=ddp_model,
            # train_loader=train_loader,
            # validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device(LOCAL_RANK),
            vocab=vocab,
            scaler=torch.cuda.amp.GradScaler(enabled=args.fp16)
        )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        total_elapsed = time.time() - total_start_time
        printgpu(f"Exiting after {str(timedelta(seconds=total_elapsed))} hours.")


if __name__ == "__main__":
    main()