import json
import argparse
import warnings
from pathlib import Path
from typing import List

def get_parser():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser

def add_arguments(parser: argparse.ArgumentParser):    
    data_input = parser.add_mutually_exclusive_group(required=True)
    data_input.add_argument(
        "-d",
        "--data-source",
        type=str,
        help="The path to the preprocessed .parquet file.",
    )
    data_input.add_argument(
        "-ds",
        "--data-sources",
        nargs='+',
        help="An array of paths to multiple preprocessed .parquet files.",
    )
    data_input.add_argument(
        "--tissues",
        nargs="+",
        help="An array of tissue names to load. The data will be loaded from the path specified in --data_tissue_path. " \
        "Depending whether --streaming is set, the tissue data will be expected to be in a single .parquet file in the same directory or in multiple sharded .parquet files (shard_***.parquet) in a subdirectory with the corresponding tissue name.",
    )
    parser.add_argument(
        "--data-tissue-path",
        type=str,
        help="The base path where the .parquet files for the tissues are stored. See --tissues for more details.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="The directory to use for caching streaming datasets. If not provided, a 'cache' subdirectory will be created in the same directory as the data files.",
    )
     # settings for saving and loading models
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=True,
        help="The directory to save the trained model and the results.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="The directory containing the model and configs to load and continue training. "
        "The following files are expected in the directory: checkpoint-{epoch}-{step}.pt, vocab.json, args.json. "
        "The last checkpoint will automatically be selected and loaded.",
    )

    # settings for data
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=None,
        help="The number of highly variable genes. If set to 0, will use all genes. "
        "Default is None, which will determine the n_hvg automatically.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float_in_range_0_1,
        default=0.0,
        help="The ratio of the validation set out of the total data. Expects a float between 0 and 1. Default is 0.0.",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float_in_range_0_1,
        default=1.0,
        help="The ratio of data to use for training. Expects a float between 0 and 1. "
        "Useful for debugging with a smaller dataset. Default is 1.0 (use all data).",
    )
    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=False,
        help="Whether to enable streaming data loading. Default is False."
        "Expects a boolean flag like True/False or yes/no (case insensitive).",
    )
    parser.add_argument(
        "--interleaved",
        type=str2bool,
        default=False,
        help="Whether the input data is interleaved across tissues. Only used if --streaming is True. Default is False."
        "Expects a boolean flag like True/False or yes/no (case insensitive).",
    )

    # settings for tokenizer
    parser.add_argument(
        "--pad-token",
        type=str,
        default="<pad>",
        help="The token to use for padding. Default is <pad>.",
    )
    parser.add_argument(
        "--input-style",
        type=str,
        choices=["normed_raw", "log1p", "binned"],
        default="binned",
        help="The style of the input data. Default is binned.",
    )
    parser.add_argument(
        "--input-emb-style",
        type=str,
        choices=["category", "continuous", "scaling"],
        default="continuous",
        help="The style of the input embedding. Default is continuous.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=51,
        help="The number of bins to use for the binned input style. Default is 51.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1000,
        help="The maximum length of the sequence. Default is 1000. The actual used "
        "max length would be the minimum of this value and the length of the longest "
        "sequence in the data.",
    )
    # omit the args for MLM and MVC, will always use them by default
    parser.add_argument(
        "--training-tasks",  #  choices of "mlm", "gen", "both"
        type=str,
        default="both",
        choices=["pcpt", "gen", "both"],
        help="The tasks to use for training. pcpt: perception training with maked token "
        "learning. gen: generation. Default is both.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.40,
        help="The ratio of masked values in the training data. Default is 0.40. This"
        "value will be ignored if --training-tasks is set to gen or both.",
    )
    parser.add_argument(
        "--trunc-by-sample",
        action="store_true",
        help="Whether to truncate the input by sampling rather than cutting off if "
        "sequence length > max_seq_length. Default is False.",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to the vocabulary file.",
    )
    # settings for training
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="The local rank of the process for using the torch.distributed.launch "
        "utility. Will be -1 if not running in distributed model.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="The local rank of the process for using the torch.distributed.launch "
        "utility. Will be -1 if not running in distributed model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size for training. Default is 32.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="The batch size for evaluation. Default is 32.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="The learning rate for training. Default is 1e-3.",
    )
    parser.add_argument(
        "--scheduler-interval",
        type=int,
        default=100,
        help="The interval iterations for updating the learning rate. Default is 100. "
        "This will only be used when warmup-ratio is 0.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.99,
        help="The factor for updating the learning rate. Default is 0.99. "
        "This will only be used when warmup-ratio is 0.",
    )
    parser.add_argument(
        "--warmup-ratio-or-steps",
        type=float,
        default=0.1,
        help="The ratio of warmup steps out of the total training steps. Default is 0.1. "
        "If warmup-ratio is above 0, will use a cosine scheduler with warmup. If "
        "the value is above 1, will use it as the number of warmup steps.",
    )
    parser.add_argument(
        "--grad-accu-steps",
        type=int,
        default=1,
        help="The number of gradient accumulation steps. Default is 1.",
    )
    parser.add_argument(
        "--no-cls",
        action="store_true",
        help="Whether to deactivate the classification loss. Default is False.",
    )
    parser.add_argument(
        "--no-cce",
        action="store_true",
        help="Whether to deactivate the contrastive cell embedding objective. "
        "Default is False.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to train in automatic mixed precision. Default is False.",
    )
    parser.add_argument(
        "--fast-transformer",
        type=bool,
        default=True,
        help="Whether to use the fast transformer. Default is True.",
    )

    # settings for model
    parser.add_argument(
        "--nlayers",
        type=int,
        default=4,
        help="The number of layers for the transformer. Default is 4.",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=4,
        help="The number of heads for the transformer. Default is 4.",
    )
    parser.add_argument(
        "--embsize",
        type=int,
        default=64,
        help="The embedding size for the transformer. Default is 64.",
    )
    parser.add_argument(
        "--d-hid",
        type=int,
        default=64,
        help="dimension of the feedforward network model in the transformer. "
        "Default is 64.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="The dropout rate. Default is 0.2.",
    )
    parser.add_argument(
        "--n-layers-cls",
        type=int,
        default=3,
        help="The number of layers for the classification network, including the "
        "output layer. Default is 3.",
    )

    # settings for logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="The interval for logging. Default is 100.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="The interval for saving the model. Default is 1000.",
    )
    parser.add_argument(
        "--separate-gpu-log-files",
        action="store_true",
        help="Whether to create separate log files for each GPU in distributed "
        "training. Default is False.",
    )


def _parse_args() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
        save_dir = args.save_dir
        args = load_args_from_model_dir(checkpoint_dir)
        args.checkpoint_dir = checkpoint_dir
        args.save_dir = save_dir
    return args


def load_args_from_model_dir(model_dir: str) -> argparse.Namespace:
    args_path = Path(model_dir) / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Could not find args.json in {model_dir}.")

    with open(args_path, "r") as f:
        args_dict = json.load(f)

    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)

    return args


def validate_args(args: argparse.Namespace):
    if args.tissues is not None and args.data_tissue_path is None:
        raise ValueError("If --tissues is provided, --data-tissue-path must also be provided.")
    if args.tissues is None and args.data_tissue_path is not None:
        warnings.warn("--data-tissue-path is provided but --tissues is not. --data-tissue-path will be ignored.", UserWarning)
    if args.tissues is None and args.interleaved is not None:
        warnings.warn("--interleaved is provided but --tissues is not. --interleaved will be ignored.", UserWarning)


def get_datapaths(args: argparse.Namespace) -> List[Path]:
    """
    Get the data sources from the command line arguments.
    We can specify either one or multiple data_sources, so this function returns a uniform data structure.
    """
    datapaths = []
    # Check which data source argument was provided
    if args.data_source is not None:
        # Single data source provided
        datapaths = [Path(args.data_source)]  # Convert to list for uniform processing

    elif args.data_sources is not None:
        # Multiple data sources provided
        datapaths = [Path(ds) for ds in args.data_sources]  # Convert to list of Paths

    elif args.tissues is not None:
        if args.data_tissue_path is None:
            raise ValueError("If --tissues is provided, --data-tissue-path must also be provided.")
        if args.interleaved:    
            tissue_dir = "-".join(sorted(args.tissues))
            interleaved_path = Path(f"{args.data_tissue_path}/{tissue_dir}")
            if not interleaved_path.is_dir():
                raise ValueError(f"Expected directory for interleaved tissues at {interleaved_path}, but it does not exist or is not a directory.")
            datapaths = list(interleaved_path.glob("shard_*.parquet"))
        else:
            if args.streaming:
                # raise DeprecationWarning("Streaming mode with --tissues is deprecated. Please use interleaved mode instead.")
                for tissue in args.tissues:
                    tissue_path = Path(f"{args.data_tissue_path}/{tissue}")
                    if not tissue_path.is_dir():
                        raise ValueError(f"Expected directory for tissue '{tissue}' at {tissue_path}, but it does not exist or is not a directory.")
                    shards = list(tissue_path.glob("shard_*.parquet"))
                    datapaths.extend(shards)
            else:        
                for tissue in args.tissues:
                    tissue_file = Path(f"{args.data_tissue_path}/{tissue}.parquet")
                    if not tissue_file.is_file():
                        raise ValueError(f"Expected file for tissue '{tissue}' at {tissue_file}, but it does not exist.")
                    datapaths.append(tissue_file)
        datapaths = [str(path) for path in datapaths]
        return datapaths



def float_in_range_0_1(value: str) -> float:
    """Custom argparse type to ensure a float is in the range (0.0, 1.0)."""
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float")

    if fvalue < 0.0 or fvalue >= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not in the range [0.0, 1.0)")

    return fvalue


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')