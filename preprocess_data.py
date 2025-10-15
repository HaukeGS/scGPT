
import os
import argparse
import json
from pathlib import Path
from typing import List
from datasets import Dataset, load_dataset, interleave_datasets
from torch.utils.data import DataLoader
import scgpt as scg
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from scgpt.tokenizer import GeneVocab


class DataPreprocessor():
    def __init__(self, args: argparse.Namespace):
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.tissues = args.tissues
        self.dataset_paths = [Path(f"{args.input_dir}/{tissue}.parquet") for tissue in self.tissues]
        self.batch_size = args.batch_size
        self.seed = args.seed

        self.vocab_path = args.vocab_path
        self.pad_token = args.pad_token

        self.max_seq_len = args.max_seq_len
        self.pad_value = args.pad_value
        self.input_style = args.input_style
        self.mask_ratio = args.mask_ratio
        self.mask_value = args.mask_value
        self.trunc_by_sample = args.trunc_by_sample
        self.training_tasks = args.training_tasks
        self.special_tokens = args.special_tokens



    def get_datasets(self) -> List[Dataset]:
        datasets = [load_dataset("parquet", data_files=str(path), split="train", streaming=True) for path in self.dataset_paths]
        with open("/home/hauke.schuele/total_dataset_sample_counts.json", "r") as f:
            tissue_counts = json.load(f)
            lengths = [tissue_counts[tissue] for tissue in self.tissues]
        total = sum(lengths)
        weights = [length / total for length in lengths]
        print(f"Dataset lengths: {lengths}, total: {total}, weights: {weights}, weight sum: {sum(weights)}")
        interleaved = interleave_datasets(
            datasets,
            probabilities=weights,
            seed=self.seed
        )
        interleaved = interleaved.with_format("torch")
        return interleaved


    def get_vocabulary(self) -> GeneVocab:
        """
        Load the vocabulary from the specified path.
        If the special tokens are not in the vocabulary, add them.
        """
        vocab = GeneVocab.from_file(self.vocab_path)
        for s in self.special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        return vocab



    def get_collator(self, vocab: GeneVocab) -> scg.DataCollator:
        """
        Create a data collator for the dataloader with provided arguments.
        """
        collator = scg.DataCollator(
            do_padding=True if self.max_seq_len is not None else False,
            pad_token_id=vocab[self.pad_token],
            pad_value=self.pad_value,
            do_mlm=True,
            do_binning=True if self.input_style == "binned" else False,
            mlm_probability=self.mask_ratio,
            mask_value=self.mask_value,
            max_length=self.max_seq_len,
            sampling=self.trunc_by_sample,
            data_style=self.training_tasks,
        )
        return collator


    def get_dataloader(self, dataset: Dataset, collator: scg.DataCollator) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            num_workers=1,
            pin_memory=False,
        )


    def write_to_disk(self, dataloader: DataLoader):
        schema = pq.read_schema(f"{self.dataset_paths[0]}")
        output_file = "-".join([f"{tissue}" for tissue in self.tissues])
        temp_output_file = self.output_dir / f"{output_file}.incomplete"
        writer = pq.ParquetWriter(temp_output_file, schema, compression="zstd")
        for batch in tqdm(dataloader):
            table = pa.Table.from_pydict(batch)
            writer.write_table(table)

        if writer:
            writer.close()
            temp_output_file.rename(temp_output_file.with_suffix(".parquet"))


    def run(self):
        interleaved_dataset = self.get_datasets()
        vocab = self.get_vocabulary()
        collator = self.get_collator(vocab)
        dataloader = self.get_dataloader(interleaved_dataset, collator)
        self.write_to_disk(dataloader)


def get_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess data for scGPT training")
    # Dataloader parameters
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input parquet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output parquet files")
    parser.add_argument("--tissues", type=str, nargs='+', required=True, help="List of tissue names corresponding to parquet files")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # Vocabulary parameters
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--pad-token", type=str, default="<pad>", help="Padding token for the vocabulary. Default is <pad>.")
    # Collator parameters
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1000,
        help="The maximum length of the sequence. Default is 1000. The actual used "
        "max length would be the minimum of this value and the length of the longest "
        "sequence in the data.",
    )
    parser.add_argument(
        "--pad-value",
        type=int,
        default=-2,
        help="The value to use for padding. Default is -2.",
    )
    parser.add_argument(
        "--input-style",
        type=str,
        choices=["normed_raw", "log1p", "binned"],
        default="binned",
        help="The style of the input data. Default is binned.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.40,
        help="The ratio of masked values in the training data. Default is 0.40. This"
        "value will be ignored if --training-tasks is set to gen or both.",
    )
    parser.add_argument(
        "--mask-value",
        type=int,
        default=-1,
        help="The value to use for masking. Default is -1.",
    )
    parser.add_argument(
        "--trunc-by-sample",
        action="store_true",
        help="Whether to truncate the input by sampling rather than cutting off if "
        "sequence length > max_seq_length. Default is False.",
    )
    parser.add_argument(
        "--training-tasks",  #  choices of "mlm", "gen", "both"
        type=str,
        default="both",
        choices=["pcpt", "gen", "both"],
        help="The tasks to use for training. pcpt: perception training with maked token "
        "learning. gen: generation. Default is both.",
    )

    args = parser.parse_args()
    args.special_tokens = [args.pad_token, "<cls>", "<eoc>"]
    print(args)
    return args



def main():
    args = get_arguments()
    preprocessor = DataPreprocessor(args)
    preprocessor.run()

if __name__ == "__main__":
    main()