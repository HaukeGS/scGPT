import argparse
from pathlib import Path
from typing import List

def get_parser():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser

def add_arguments(parser: argparse.ArgumentParser):    
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "-d",
        "--data-source",
        type=str,
        help='The name of the data source (currently support "scvi" datasets), or the '
        "path to the data file.",
    )
    data_group.add_argument(
        "-ds",
        "--data-sources",
        nargs='+',
        help='An array of paths to datasources. Expects to find a preprocessed cls_prefix_data.parquet file in each directory',
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=True,
        help="The directory to save the trained model and the results.",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="The directory containing the model and configs to load and continue training.",
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
        default=0.1,
        help="The ratio of the validation set out of the total data. Expects a float between 0 and 1. Default is 0.1.",
    )

    parser.add_argument(
        "--grad-accu-steps",
        type=int,
        default=1,
        help="The number of gradient accumulation steps. Default is 1.",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float_in_range_0_1,
        default=1.0,
        help="The ratio of data to use for training. Expects a float between 0 and 1. "
        "Useful for debugging with a smaller dataset. Default is 1.0 (use all data).",
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

def get_datapaths(args: argparse.Namespace) -> List[Path]:
    """
    Get the data sources from the command line arguments.
    We can specify either one or multiple data_sources, so this function returns a uniform data structure.
    """
    data_paths = []
    # Check which data source argument was provided
    if args.data_source is not None:
        # Single data source provided
        data_paths = [Path(args.data_source)]  # Convert to list for uniform processing

    elif args.data_sources is not None:
        # Multiple data sources provided
        data_paths = [Path(ds) for ds in args.data_sources]  # Convert to list of Paths
        
    else:
        # This shouldn't happen due to required=True, but good practice
        raise ValueError("No data source provided")

    return data_paths


def float_in_range_0_1(value: str) -> float:
    """Custom argparse type to ensure a float is in the range (0.0, 1.0)."""
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float")

    if fvalue <= 0.0 or fvalue >= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not in the range (0.0, 1.0)")

    return fvalue