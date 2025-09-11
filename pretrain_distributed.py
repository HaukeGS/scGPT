

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
            

def dump_debug(output: str) -> None:
    """
    Dump the debug information to a text file in the save directory.
    """
    if is_master_gpu():
        with open(SAVE_DIR / "debug.txt", "a") as f:
            f.write(output + "\n")


def create_dataset(data_paths: List[Path]) -> Dataset:
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
        printmaster(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
        example_count += len(raw_dataset)
    printmaster(f"Total examples across all datasets: {example_count}")
    merged_dataset = concatenate_datasets(datasets)
    merged_dataset = merged_dataset.with_format("torch")
    printmaster(f"dataset columns: {merged_dataset.column_names}")
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
    printmaster(f"Train loader length for one GPU: {len(train_loader)}")
    printmaster(f"Train loader length for all GPUs: {len(train_loader) * WORLD_SIZE}")
    printmaster(f"Validation loader length for all GPUs: {len(validation_loader)}")
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
    ) -> transformers.get_scheduler:
    """
    Get the learning rate scheduler.
    """
    if warmup_ratio_or_steps > 0:
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
    ) -> None:
    """
    Train the model for the specified number of epochs.
    """
    best_val_loss = float("inf")
    writer = SummaryWriter(log_dir=SAVE_DIR / "tensorboard")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
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

            # optimizer.zero_grad()
            # printmaster(f"total loss before backward: {total_loss.item()}")
            # total_loss.backward()
            # printmaster(f"total loss after backward: {total_loss.item()}")
            # running_loss += total_loss.item()
            # optimizer.step()
            # scheduler.step()

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            

            if is_master_gpu() and global_iter % log_interval == 0:
                writer.add_scalar("loss/mse", loss_mse, global_iter)
                writer.add_scalar("loss/mvc", loss_mvc, global_iter)
                writer.add_scalar("loss/gen", loss_gen, global_iter)
                writer.add_scalar("loss/total", total_loss, global_iter)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_iter)
                writer.add_scalar("running_loss", running_loss / (i + 1), global_iter)



            if i >= 4000:
                break
    writer.close()
    printmaster("Training complete.")




def average_across_processes(tensor):
    """Average a tensor across all processes"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()



# def train(
    #     model: DDP, 
    #     train_loader: DataLoader, 
    #     valid_loader: DataLoader,
    #     criterion: nn.Module,
    #     criterion_cls: nn.Module,
    #     optimizer: torch.optim.Optimizer,
    #     scheduler: transformers.get_scheduler,
    #     epoch: int,
    #     log_interval: int,
    #     save_interval: int,
    #     device: torch.device,
    #     vocab: GeneVocab,
    #     pad_token: str,
    #     mask_value: float,
    #     fp16_enabled: bool,
    #     scaler: torch.cuda.amp.GradScaler,
    #     grad_accu_steps: int,
    # ) -> None:
    # """
    # Train the model for one epoch.
    # """
    # writer = SummaryWriter(log_dir=SAVE_DIR / "tensorboard")
    # model.train()
    # total_loss, total_mse, total_cls, total_gen, total_mvc = 0.0, 0.0, 0.0, 0.0, 0.0
    # total_error = 0.0
    # log_interval = log_interval
    # start_time = time.time()

    # num_batches = len(train_loader)
    # for batch, data_dict in enumerate(train_loader):
    #     global_iter = epoch * num_batches + batch

    #     data_dict = {k: v.to(device) for k, v in data_dict.items()}
    #     # print(f"\n+++ data_dict keys: {data_dict.keys()}")
    #     if USE_GENERATIVE_TRAINING:
    #         pcpt_gene = data_dict["pcpt_gene"]
    #         pcpt_expr = data_dict["pcpt_expr"]
    #         pcpt_key_padding_mask = pcpt_gene.eq(vocab[pad_token])
    #         gen_gene = data_dict["gen_gene"]
    #         gen_expr_target = target_values = data_dict["gen_expr_target"]
    #         gen_key_padding_mask = gen_gene.eq(vocab[pad_token])
    #     else:
    #         input_gene_ids = data_dict["gene"]
    #         input_values = data_dict["masked_expr"]
    #         target_values = data_dict["expr"]
    #         src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

    #     with torch.cuda.amp.autocast(enabled=fp16_enabled):
    #         if USE_GENERATIVE_TRAINING:
    #             output_dict = model(
    #                 pcpt_gene,
    #                 pcpt_expr,
    #                 pcpt_key_padding_mask,
    #                 gen_gene,
    #                 gen_key_padding_mask,
    #                 CLS=USE_CLS,
    #                 MVC=MVC,
    #                 generative_training=True,
    #             )
    #             gen_expr_preds = output_values = output_dict["gen_preds"]

    #             positions_to_match = ~gen_key_padding_mask
    #             loss = loss_mse = criterion(
    #                 gen_expr_preds, gen_expr_target, positions_to_match
    #             )
    #             writer.add_scalar("train/mse", loss_mse, global_iter)
    #             if MVC:
    #                 loss_mvc = criterion(
    #                     output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
    #                     gen_expr_target,
    #                     positions_to_match,
    #                 )
    #                 loss = loss + loss_mvc
    #                 writer.add_scalar("train/mvc", loss_mvc, global_iter)
    #         else:
    #             output_dict = model(
    #                 input_gene_ids,
    #                 input_values,
    #                 src_key_padding_mask=src_key_padding_mask,
    #                 CLS=USE_CLS,
    #                 CCE=USE_CCE,  # TODO: move these flags to model's attributes
    #                 MVC=MVC,
    #                 generative_training=False,
    #             )
    #             output_values = output_dict["mlm_output"]

    #             positions_to_match = input_values.eq(
    #                 mask_value
    #             )  # the postions to predict
    #             loss = loss_mse = criterion(
    #                 output_values, target_values, positions_to_match
    #             )
    #             writer.add_scalar("train/mse", loss_mse, global_iter)
    #             if USE_CLS:
    #                 target_labels = data_dict["celltypes"]
    #                 loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
    #                 loss = loss + loss_cls
    #                 writer.add_scalar("train/cls", loss_cls, global_iter)
    #             if USE_CCE:
    #                 loss_cce = 10 * output_dict["loss_cce"]
    #                 loss = loss + loss_cce
    #                 writer.add_scalar("train/cce", loss_cce, global_iter)
    #             if MVC:
    #                 loss_mvc = criterion(
    #                     output_dict["mvc_output"], target_values, positions_to_match
    #                 )
    #                 loss = loss + loss_mvc
    #                 writer.add_scalar("train/mvc", loss_mvc, global_iter)
    #         writer.add_scalar("train/loss", loss, global_iter)

    #         if USE_GENERATIVE_TRAINING and global_iter > 1000:
    #             previous_cell_embs = output_dict["cell_emb"].detach()
    #             preds = model(
    #                 pcpt_gene,
    #                 pcpt_expr,
    #                 pcpt_key_padding_mask,
    #                 gen_gene,
    #                 gen_key_padding_mask,
    #                 CLS=False,
    #                 MVC=False,
    #                 input_cell_emb=previous_cell_embs,
    #                 generative_training=True,
    #             )["gen_preds"]
    #             loss_gen = criterion(preds, gen_expr_target, positions_to_match)
    #             loss = loss + loss_gen
    #             writer.add_scalar("train/gen", loss_gen, global_iter)

    #             # TODO: try this choice of using a separate backprop
    #             # # this part is for the choice of using a separate backprop
    #             # model.zero_grad()
    #             # scaler.scale(loss_gen).backward()
    #             # scaler.unscale_(optimizer)
    #             # torch.nn.utils.clip_grad_norm_(
    #             #     model.parameters(),
    #             #     1.0,
    #             #     error_if_nonfinite=False if scaler.is_enabled() else True,
    #             # )
    #             # scaler.step(optimizer)
    #             # scaler.update()

    #     if grad_accu_steps > 1:
    #         loss = loss / grad_accu_steps
    #     scaler.scale(loss).backward()
    #     scaler.unscale_(optimizer)
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #     scaler.step(optimizer)
    #     scaler.update()

    #     if grad_accu_steps > 1:
    #         if batch % grad_accu_steps == 0 or batch == num_batches - 1:
    #             scheduler.step()
    #             optimizer.zero_grad()
    #     else:
    #         scheduler.step()
    #         optimizer.zero_grad()

    #     with torch.no_grad():
    #         mre = masked_relative_error(
    #             output_values, target_values, positions_to_match
    #         )
    #         writer.add_scalar("train/mre", mre, global_iter)

    #     total_loss += loss.item()
    #     total_mse += loss_mse.item()
    #     total_cls += loss_cls.item() if USE_CLS else 0.0
    #     total_gen += loss_gen.item() if "loss_gen" in locals() else 0.0
    #     total_mvc += loss_mvc.item() if MVC else 0.0
    #     total_error += mre.item()

    #     # if args.local_rank in [0, -1] and batch % log_interval == 0 and batch > 0:
    #     if is_master_gpu() and batch % log_interval == 0 and batch > 0:
    #         # Writer logs gradients distribution
    #         for name, param in model.named_parameters():
    #             if param.requires_grad and param.grad is not None:
    #                 writer.add_histogram(name + "_grad", param.grad, global_iter)
    #                 writer.add_histogram(name + "_param", param, global_iter)

    #         # Log scalar values
    #         lr = scheduler.get_last_lr()[0]
    #         ms_per_batch = (time.time() - start_time) * 1000 / log_interval
    #         cur_loss = total_loss / log_interval
    #         cur_mse = total_mse / log_interval
    #         cur_cls = total_cls / log_interval if USE_CLS else 0.0
    #         cur_gen = total_gen / log_interval if "loss_gen" in locals() else 0.0
    #         cur_mvc = total_mvc / log_interval if MVC else 0.0
    #         cur_error = total_error / log_interval
    #         # ppl = math.exp(cur_loss)
    #         logger.info(
    #             f"| epoch {epoch:1d} | {batch:4d}/{num_batches:3d} batches | "
    #             f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
    #             f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
    #             + (f"cls {cur_cls:05.2f} | " if USE_CLS else "")
    #             + (f"gen {cur_gen:05.2f} |" if "loss_gen" in locals() else "")
    #             + (f"mvc {cur_mvc:05.2f} |" if MVC else "")
    #         )
    #         writer.add_scalar("lr", lr, global_iter)
    #         writer.flush()

    #         # if (batch > 1000):
    #         #     print(f"Reached batch goal of 1000, stopping training.")
    #         #     sys.exit("Stopping training after 1000 batches.")

    #         total_loss = 0
    #         total_mse = 0
    #         total_cls = 0
    #         total_gen = 0
    #         total_mvc = 0
    #         total_error = 0
    #         start_time = time.time()

    #     # immediately eval and save
    #     # if batch % save_interval == 0 and batch > 0:
    #     #     eval_and_save(model, valid_loader, global_iter)
    #     #     model.train()  # important, reset to train mode
    #     if is_master_gpu() and (batch + 1) % save_interval == 0:
    #         mse_loss, mre_loss = evaluate(
    #             model=model,
    #             valid_loader=valid_loader,
    #             device=device,
    #             vocab=vocab,
    #             pad_token=pad_token,
    #             fp16_enabled=fp16_enabled,
    #             mask_value=mask_value,
    #             criterion=criterion
    #         ).values()

    #         writer.add_scalar("valid/mse", mse_loss, global_iter)
    #         writer.add_scalar("valid/mre", mre_loss, global_iter)
    #         writer.flush()


def train(
        model: nn.Module, 
        train_loader: DataLoader, 
        epoch: int,
        args: argparse.Namespace,
        device: torch.device,
        vocab: GeneVocab,
        writer: SummaryWriter,
        criterion: nn.Module,
        criterion_cls: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: transformers.get_scheduler,
        scaler: torch.cuda.amp.GradScaler,
        valid_loader: DataLoader,
    ) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    logger.info(model)
    logger.info(train_loader)
    logger.info(criterion)
    logger.info(optimizer)
    logger.info(scheduler)
    logger.info(scaler)
    total_loss, total_mse, total_cls, total_gen, total_mvc = 0.0, 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(train_loader)
    logger.info(f"Number of batches: {num_batches}")
    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        logger.info(f"data_dict keys: {data_dict.keys()}")
        # print(f"\n+++ data_dict keys: {data_dict.keys()}")
        if USE_GENERATIVE_TRAINING:
            pcpt_gene = data_dict["pcpt_gene"]
            pcpt_expr = data_dict["pcpt_expr"]
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
            gen_gene = data_dict["gen_gene"]
            gen_expr_target = target_values = data_dict["gen_expr_target"]
            gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])
        else:
            input_gene_ids = data_dict["gene"]
            input_values = data_dict["masked_expr"]
            target_values = data_dict["expr"]
            src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

        logger.info(f"data_dict keys: {data_dict.keys()}")
        with torch.cuda.amp.autocast(enabled=args.fp16):
            if USE_GENERATIVE_TRAINING:
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
                logger.info(f"output_dict keys: {output_dict.keys()}")
                gen_expr_preds = output_values = output_dict["gen_preds"]
                logger.info(f"gen_expr_preds shape: {gen_expr_preds.shape}")
                positions_to_match = ~gen_key_padding_mask
                logger.info(f"positions_to_match shape: {positions_to_match.shape}")
                loss = loss_mse = criterion(
                    gen_expr_preds, gen_expr_target, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse, global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
                        gen_expr_target,
                        positions_to_match,
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc, global_iter)
            else:
                logger.info(f"WARNING: Using non-generative training!")
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=USE_CLS,
                    CCE=USE_CCE,  # TODO: move these flags to model's attributes
                    MVC=MVC,
                    generative_training=False,
                )
                output_values = output_dict["mlm_output"]

                positions_to_match = input_values.eq(
                    args.mask_value
                )  # the postions to predict
                loss = loss_mse = criterion(
                    output_values, target_values, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse, global_iter)
                if USE_CLS:
                    target_labels = data_dict["celltypes"]
                    loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
                    loss = loss + loss_cls
                    writer.add_scalar("train/cls", loss_cls, global_iter)
                if USE_CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    writer.add_scalar("train/cce", loss_cce, global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, positions_to_match
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc, global_iter)
            writer.add_scalar("train/loss", loss, global_iter)

            if USE_GENERATIVE_TRAINING and global_iter > 1000:
                previous_cell_embs = output_dict["cell_emb"].detach()
                logger.info(f"previous_cell_embs shape: {previous_cell_embs.shape}")
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
                logger.info(f"preds shape: {preds.shape}")
                loss_gen = criterion(preds, gen_expr_target, positions_to_match)
                loss = loss + loss_gen
                writer.add_scalar("train/gen", loss_gen, global_iter)

                # TODO: try this choice of using a separate backprop
                # # this part is for the choice of using a separate backprop
                # model.zero_grad()
                # scaler.scale(loss_gen).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(),
                #     1.0,
                #     error_if_nonfinite=False if scaler.is_enabled() else True,
                # )
                # scaler.step(optimizer)
                # scaler.update()

        if args.grad_accu_steps > 1:
            loss = loss / args.grad_accu_steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if args.grad_accu_steps > 1:
            if batch % args.grad_accu_steps == 0 or batch == num_batches - 1:
                scheduler.step()
                optimizer.zero_grad()
        else:
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            mre = masked_relative_error(
                output_values, target_values, positions_to_match
            )
            logger.info(f"output_values shape: {output_values.shape}")
            logger.info(f"target_values shape: {target_values.shape}")
            logger.info(f"mre: {mre.item()}")
            writer.add_scalar("train/mre", mre, global_iter)

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cls += loss_cls.item() if USE_CLS else 0.0
        total_gen += loss_gen.item() if "loss_gen" in locals() else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_error += mre.item()
        if args.local_rank in [0, -1] and batch % log_interval == 0 and batch > 0:
            # Writer logs gradients distribution
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(name + "_grad", param.grad, global_iter)
                    writer.add_histogram(name + "_param", param, global_iter)

            # Log scalar values
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if USE_CLS else 0.0
            cur_gen = total_gen / log_interval if "loss_gen" in locals() else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:1d} | {batch:4d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"cls {cur_cls:05.2f} | " if USE_CLS else "")
                + (f"gen {cur_gen:05.2f} |" if "loss_gen" in locals() else "")
                + (f"mvc {cur_mvc:05.2f} |" if MVC else "")
            )
            writer.add_scalar("lr", lr, global_iter)
            writer.flush()

            if (batch > 4000):
                print(f"Reached batch goal of 1000, stopping training.")
                sys.exit("Stopping training after 1000 batches.")

            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_gen = 0
            total_mvc = 0
            total_error = 0
            start_time = time.time()

        # immediately eval and save
        if batch % args.save_interval == 0 and batch > 0:
            eval_and_save(model, valid_loader, global_iter)
            model.train()  # important, reset to train mode


def eval_and_save(model: DDP, valid_loader: DataLoader, global_iter: int) -> None:
    pass


def evaluate(
        model: DDP, 
        valid_loader: DataLoader, 
        device: torch.device, 
        vocab: GeneVocab,
        pad_token: str,
        fp16_enabled: bool,
        mask_value: float,
        criterion: nn.Module,
    ) -> Dict[str, torch.Tensor]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    with torch.no_grad():
        for data_dict in valid_loader:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            if USE_GENERATIVE_TRAINING:
                pcpt_gene = data_dict["pcpt_gene"]
                pcpt_expr = data_dict["pcpt_expr"]
                pcpt_key_padding_mask = pcpt_gene.eq(vocab[pad_token])
                gen_gene = data_dict["gen_gene"]
                gen_expr_target = target_values = data_dict["gen_expr_target"]
                gen_key_padding_mask = gen_gene.eq(vocab[pad_token])
            else:
                input_gene_ids = data_dict["gene"]
                input_values = data_dict["masked_expr"]
                target_values = data_dict["expr"]
                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=fp16_enabled):
                if USE_GENERATIVE_TRAINING:
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
                    gen_expr_preds = output_values = output_dict["gen_preds"]

                    positions_to_match = ~gen_key_padding_mask
                else:
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        generative_training=False,
                    )
                    output_values = output_dict["mlm_output"]
                    positions_to_match = input_values.eq(mask_value)

                loss = criterion(output_values, target_values, positions_to_match)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, positions_to_match
            ).item()
    total_loss = total_loss / len(valid_loader)
    total_error = total_error / len(valid_loader)
    return {
        "mse": torch.tensor(total_loss, device=device, dtype=torch.float),
        "mre": torch.tensor(total_error, device=device, dtype=torch.float),
    }



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
    SAVE_DIR = Path(args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)
    USE_GENERATIVE_TRAINING = True if args.training_tasks in ["gen", "both"] else False
    SPECIAL_TOKENS = [args.pad_token, "<cls>", "<eoc>"]
    USE_CLS = not args.no_cls
    USE_CCE = not args.no_cce
    MVC = True

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
        # Load data
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
        )
        dist.barrier()
        time.sleep(2)
        # pretrain the model
        # pretrain(
        #     model=ddp_model,
        #     train_loader=train_loader,
        #     validation_loader=validation_loader,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     num_epochs=1,
        #     log_interval=args.log_interval,
        #     save_interval=args.save_interval,
        #     device=torch.device(LOCAL_RANK),
        #     vocab=vocab,
        #     pad_token=args.pad_token,
        #     mask_value=args.mask_value,
        #     fp16_enabled=args.fp16,
        #     scaler=torch.cuda.amp.GradScaler(enabled=args.fp16),
        # )
        train(
            model=ddp_model,
            train_loader=train_loader,
            epoch=0,
            args=args,
            device=torch.device(LOCAL_RANK),
            vocab=vocab,
            writer=SummaryWriter(log_dir=SAVE_DIR / "tensorboard"),
            criterion=criterion,
            criterion_cls=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=torch.cuda.amp.GradScaler(enabled=args.fp16),
            valid_loader=validation_loader,
        )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        printgpu(f"Exiting...")


if __name__ == "__main__":
    main()