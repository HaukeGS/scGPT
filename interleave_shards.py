
import math
import os
import argparse
import json
from pathlib import Path
from typing import List
from datasets import Dataset, load_dataset, interleave_datasets
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm



class DataPreprocessor():
    def __init__(
        self,
        tissues: List[str],
        input_dir: Path,
        output_dir: Path,
        n_shards: int,
    ):
        self.tissues = tissues
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_shards = n_shards
        self.dataset_paths = [
            Path(f"{self.input_dir}/{tissue}.parquet")
            for tissue in tissues
        ]

    def get_total_samples(self) -> int:
        total_sample_count = 0
        with open("/home/hauke.schuele/total_dataset_sample_counts.json", "r") as f:
            tissue_counts = json.load(f)
            for tissue in self.tissues:
                total_sample_count += tissue_counts[tissue]
        return total_sample_count
    

    def get_shardsize(self) -> int:
        return math.ceil(self.get_total_samples() / self.n_shards)
    

    def get_datasets(self) -> Dataset:
        """Load and interleave datasets."""
        datasets = []
        sample_counts = []
        
        for tissue in self.tissues:
            path = Path(f"{self.input_dir}/{tissue}.parquet")
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            dataset = load_dataset(
                "parquet",
                data_files=str(path),
                split="train",
                streaming=True,  # Use streaming to avoid loading all into memory
            )
            datasets.append(dataset)

            with open("/home/hauke.schuele/total_dataset_sample_counts.json", "r") as f:
                tissue_counts = json.load(f)
                sample_count = tissue_counts[tissue]
                sample_counts.append(sample_count)
        total = sum(sample_counts)
        probabilities = [count / total for count in sample_counts]
        interleaved = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=42,
            stopping_strategy="first_exhausted"
        )
        
        return interleaved

    def write_to_disk(self, dataset: Dataset):
        """Write streaming dataset to sharded parquet files."""
        output_file = "-".join([f"{tissue}" for tissue in self.tissues])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Write in batches/shards
        total_samples = self.get_total_samples()
        num_rows_per_shard = self.get_shardsize()
        num_shards = self.n_shards
        batch_size = min(10_000, num_rows_per_shard)  # Adjust batch size as needed
        batches_iter = dataset.iter(batch_size=batch_size)

        schema = pq.read_schema(self.dataset_paths[0])

        def create_writer():
            writer = pq.ParquetWriter(self.output_dir / f"{output_file}_{shard:03d}.incomplete", schema, compression="zstd")
            return writer
        
        writer = None
        shard=0
        temp_output_file = self.output_dir / f"{output_file}_{shard:03d}.incomplete"
        rows_written = 0
        with tqdm(total=total_samples, desc="Interleaving and sharding dataset") as pbar:
            for batch_dict in batches_iter:
                if writer is None:
                    writer = create_writer()

                batch = pa.Table.from_pydict(batch_dict, schema=schema)

                offset_in_batch = 0
                remaining_in_batch = batch.num_rows

                # Write the batch in slices so we never exceed the shard size
                while remaining_in_batch > 0:
                    remaining_in_shard = num_rows_per_shard - rows_written if shard < num_shards - 1 else remaining_in_batch
                    rows_to_write = min(remaining_in_batch, remaining_in_shard)

                    if rows_to_write > 0:
                        slice_to_write = batch.slice(offset_in_batch, rows_to_write)
                        writer.write_table(slice_to_write)
                        rows_written += rows_to_write
                        offset_in_batch += rows_to_write
                        remaining_in_batch -= rows_to_write
                        pbar.update(rows_to_write)

                    # If current shard is filled (for all but the last shard), roll to next shard
                    if (shard < num_shards - 1) and (rows_written == num_rows_per_shard):
                        writer.close()
                        temp_output_file.rename(self.output_dir / f"{output_file}_{shard:03d}.parquet")
                        shard += 1
                        writer = create_writer()
                        rows_written = 0
            if writer is not None:
                writer.close()
                temp_output_file.rename(self.output_dir / f"{output_file}_{shard:03d}.parquet")

    def run(self):
        interleaved_dataset = self.get_datasets()
        self.write_to_disk(interleaved_dataset)

def get_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess data for scGPT training")
    # Dataloader parameters
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input parquet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output parquet files")
    parser.add_argument("--tissues", type=str, nargs='+', required=True, help="List of tissue names corresponding to parquet files")
    parser.add_argument("--n_shards", type=int, required=True, help="Number of shards to split the dataset into")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))
    return args


def create_arguments():
    tissues = [
        "blood",
        "kidney",
        "pancreas",
        "intestine"
    ]
    input_dir = "/home/hauke.schuele/cellxgene_data/"
    output_dir = "/home/hauke.schuele/cellxgene_data_interleaved"
    n_shards = 25

    class Args(argparse.Namespace):
        def __init__(self, tissues, input_dir, output_dir, n_shards):
            self.tissues = tissues
            self.input_dir = input_dir
            self.output_dir = output_dir
            self.n_shards = n_shards

    args = Args(tissues, input_dir, output_dir, n_shards)
    return args


def main():
    args = get_arguments()
    preprocessor = DataPreprocessor(
        tissues=args.tissues,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        n_shards=args.n_shards,
    )
    preprocessor.run()

if __name__ == "__main__":
    main()