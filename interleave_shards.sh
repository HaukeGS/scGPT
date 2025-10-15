#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=interleave_shards              # A nice readable name of your job, to see it in the queue
#SBATCH --nodes=1                                     # Number of nodes to request
#SBATCH --ntasks-per-node=1                           # total number of tasks per node
#SBATCH --cpus-per-task=4                             # Number of CPUs to request
#SBATCH --output=/home/hauke.schuele/data_preprocessing/logs/%x-%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/hauke.schuele/data_preprocessing/logs/%x-%j.err   # File to which STDERR will be written
echo ""
#SBATCH --time=00:15:00              # Time limit (hh:mm:ss)

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



TISSUES=("blood" "kidney" "pancreas" "intestine")  # try to use two datasets and monitor I/O operations
DATA_TISSUE_PATH="/home/hauke.schuele/cellxgene_data/"

srun python -u scGPT_distributed/interleave_shards.py \
    --tissues "${TISSUES[@]}" \
    --input_dir "$DATA_TISSUE_PATH" \
    --output_dir "/home/hauke.schuele/cellxgene_data_interleaved" \
    --n_shards 25