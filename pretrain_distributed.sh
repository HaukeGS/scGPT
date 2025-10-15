#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=scGPT_dist_pretrain              # A nice readable name of your job, to see it in the queue
#SBATCH --nodes=2                                     # Number of nodes to request
#SBATCH --ntasks-per-node=2                           # total number of tasks per node
#SBATCH --cpus-per-task=4                             # Number of CPUs to request
#SBATCH --gres=gpu:a100:2                             # Number of GPUs to request
#SBATCH --mem-per-gpu=4GB
#SBATCH --partition=ampere
#SBATCH --output=/home/hauke.schuele/scGPT_distributed/logs/%x-%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/hauke.schuele/scGPT_distributed/logs/%x-%j.err   # File to which STDERR will be written
#SBATCH --time=00:15:00              # Time limit (hh:mm:ss)
echo ""

module load mamba
micromamba activate scgpt_manual

# SLURM parameters
master_address=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$master_address
export MASTER_PORT=$(((SLURM_JOB_ID % 100000) + 1024))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "WORLD_SIZE=$WORLD_SIZE"


# Script parameters
streaming="true"

if [ "$streaming" = "true" ]; then
    DATA_TISSUE_PATH="/home/hauke.schuele/cellxgene_data_sharded_validation/"
else
    DATA_TISSUE_PATH="/home/hauke.schuele/cellxgene_data/"
fi

# scGPT parameters
# TISSUES=("kidney")  # try to use smaller datasets for now
# TISSUES=("pan-cancer")  # try to use smaller datasets for now
TISSUES=("blood" "kidney" "pancreas" "intestine")  # try to use two datasets and monitor I/O operations
# TISSUES=("blood" "brain" "heart" "intestine" "kidney" "lung" "others" "pan-cancer" "pancreas")
DATA_SOURCES=()

for TISSUE in "${TISSUES[@]}"; do
    DATA_SOURCES+=("/data/datasets/biology/scGPT-data/preprocessed/$TISSUE/all_counts/cls_prefix_data.parquet")
done

srun python -u scGPT_distributed/pretrain_distributed.py \
    --tissues "${TISSUES[@]}" \
    --data-tissue-path "$DATA_TISSUE_PATH" \
    --epochs 1 \
    --training-tasks "both" \
    --save-dir ./save/pretrain-distributed-[$SLURM_JOB_ID]-$(date +%Y-%m-%d_%H-%M-%S) \
    --vocab-path "/data/datasets/biology/scGPT-data/preprocessed/default_census_vocab.json" \
    --save-interval 5000 \
    --batch-size 128 \
    --valid-ratio 0.04 \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 \
    --streaming "$streaming" \