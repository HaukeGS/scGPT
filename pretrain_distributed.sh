#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=scGPT_pretrain              # Gets overwritten by run_and_monitor_job.sh!
#SBATCH --nodes=1                                     # Number of nodes to request
#SBATCH --ntasks-per-node=8                           # total number of tasks per node
#SBATCH --cpus-per-task=4                             # Number of CPUs to request
#SBATCH --gres=gpu:a100:8                             # Number of GPUs to request
#SBATCH --mem-per-gpu=4GB
#SBATCH --partition=ampere
#SBATCH --output=/home/hauke.schuele/scGPT_distributed/logs/%x-%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/hauke.schuele/scGPT_distributed/logs/%x-%j.err   # File to which STDERR will be written
#SBATCH --mail-user=hauke.schuele@stud.uni-hannover.de
#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --time=10-00:00:00              # Time limit (hh:mm:ss)
echo ""

module load mamba
micromamba activate scgpt_manual
# micromamba activate scgpt_pretrain

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
interleaved="true"

# Tissue selection based on data percentage
data_percentage="$1"
if [ -z "$data_percentage" ]; then
    echo "No data percentage provided. Using default: 10%"
    data_percentage="10"
elif [ "$data_percentage" != "10" ] && [ "$data_percentage" != "50" ] && [ "$data_percentage" != "100" ]; then
    echo "Invalid data percentage provided: $data_percentage. Allowed values are 10, 50, or 100."
    echo "Using default: 10%"
    data_percentage="10"
else
    echo "Data percentage provided: $data_percentage%"
fi

if [ "$data_percentage" = "10" ]; then
    TISSUES=("heart" "lung")
elif [ "$data_percentage" = "50" ]; then
    TISSUES=("lung" "others" "pan-cancer")
elif [ "$data_percentage" = "100" ]; then
    TISSUES=("blood" "brain" "heart" "intestine" "kidney" "lung" "others" "pan-cancer" "pancreas")
fi
IFS=$'\n' TISSUES=($(printf '%s\n' "${TISSUES[@]}" | sort))
echo "Sorted tissues: ${TISSUES[@]}"


# Data paths
if [ "$interleaved" = "true" ]; then
    DATA_TISSUE_PATH="/home/hauke.schuele/cellxgene_data_interleaved/"
else
    if [ "$streaming" = "true" ]; then
        DATA_TISSUE_PATH="/home/hauke.schuele/cellxgene_data_sharded_validation/"
        # echo "Using streaming with non-interleaved data is no more supported. Exiting."
        # exit 1
    else
        DATA_TISSUE_PATH="/home/hauke.schuele/cellxgene_data/"
    fi
fi

# DATA_SOURCES=()

# for TISSUE in "${TISSUES[@]}"; do
#     DATA_SOURCES+=("/data/datasets/biology/scGPT-data/preprocessed/$TISSUE/all_counts/cls_prefix_data.parquet")
# done

srun python -u scGPT_distributed/pretrain_distributed_args.py \
    --tissues "${TISSUES[@]}" \
    --data-tissue-path "$DATA_TISSUE_PATH" \
    --epochs 3 \
    --training-tasks "both" \
    --save-dir ./save/pretrain-distributed-[$SLURM_JOB_ID]-$(date +%Y-%m-%d_%H-%M-%S) \
    --vocab-path "/data/datasets/biology/scGPT-data/preprocessed/default_census_vocab.json" \
    --save-interval 50000 \
    --log-interval 1000 \
    --batch-size 32 \
    --valid-ratio 0.04 \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 \
    --streaming "$streaming" \
    --interleaved "$interleaved" \
    --nlayers 12 \
    --nheads 8 \
    --embsize 512 \
    --d-hid 512 \
    # --checkpoint-dir "/home/hauke.schuele/save/pretrain-distributed-[256520]-2025-11-11_12-14-16" \