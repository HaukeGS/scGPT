#!/bin/bash

# SLURM parameters
master_address=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# master_port=$(id -u)
export MASTER_ADDR=$master_address
# export MASTER_PORT=$master_port
# export MASTER_PORT=51550
export MASTER_PORT=$(((SLURM_JOB_ID % 10000) + 50000))
export WORLD_SIZE=1
export SLURM_CPUS_PER_TASK=4
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "WORLD_SIZE=$WORLD_SIZE"


# Script parameters
streaming="false"
interleaved="false"

# Tissue selection based on data percentage
data_percentage="$1"
if [ -z "$data_percentage" ]; then
    echo "No data percentage provided. Using default: 10%"
    data_percentage="10"
elif [ "$data_percentage" != "10" ] && [ "$data_percentage" != "50" ] && [ "$data_percentage" != "100" ] && [ "$data_percentage" != "lung" ]; then
    echo "Invalid data percentage provided: $data_percentage. Allowed values are 10, 50, 100, or lung."
    echo "Using default: 10%"
    data_percentage="10"
else
    echo "Data percentage provided: $data_percentage%"
fi

if [ "$data_percentage" = "10" ]; then
    TISSUES=("heart" "lung")
elif [ "$data_percentage" = "50" ]; then
    TISSUES=("lung" "others" "pancreas")
elif [ "$data_percentage" = "100" ]; then
    TISSUES=("blood" "brain" "heart" "intestine" "kidney" "lung" "others" "pancreas")
elif [ "$data_percentage" = "lung" ]; then
    TISSUES=("lung")
fi
IFS=$'\n' TISSUES=($(printf '%s\n' "${TISSUES[@]}" | sort))
echo "Sorted tissues: ${TISSUES[@]}"

if [ "$2" = "moe" ]; then
    echo "MOE training enabled."
    NUM_EXPERTS=8
    K=2
    BATCH_SIZE=10
    DATA_TISSUE_PATH="/user/hauke.schuele/u26703/.project/dir.project/cellxgene-celltype-column-2023-05-15"
    CELL_TYPE_VOCAB_PATH="/user/hauke.schuele/u26703/.project/dir.project/cellxgene-celltype-column-2023-05-15/celltype_vocab.json"
    EXPERT_SPECIALIZATION="true"
else
    NUM_EXPERTS=0
    K=0
    BATCH_SIZE=60
    # BATCH_SIZE=40 
    DATA_TISSUE_PATH="/user/hauke.schuele/u26703/.project/dir.project/cellxgene-2023-05-15"
    CELL_TYPE_VOCAB_PATH=None
    EXPERT_SPECIALIZATION="false"
fi


DATA_TISSUE_PATH="/user/hauke.schuele/u26703/.project/dir.project/cellxgene-celltype-column-2023-05-15"


# Batch size: 20 for normal, 13 for moe with e=8, k=2


python -u /user/hauke.schuele/u26703/scGPT/pretrain_distributed_args.py \
    --tissues "${TISSUES[@]}" \
    --data-tissue-path "$DATA_TISSUE_PATH" \
    --epochs 6 \
    --training-tasks "both" \
    --save-dir "/user/hauke.schuele/u26703/scGPT/save/pretrain-distributed-[$SLURM_JOB_ID]-$(date +%Y-%m-%d_%H-%M-%S)" \
    --vocab-path "/user/hauke.schuele/u26703/.project/dir.project/cellxgene-celltype-column-2023-05-15/2023-05-15-vocab.json" \
    --cell-type-vocab-path $CELL_TYPE_VOCAB_PATH \
    --save-interval 50000 \
    --log-interval 1000 \
    --batch-size $BATCH_SIZE  \
    --valid-ratio 0.003 \
    --mask-ratio 0.4 \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 \
    --separate-gpu-log-files \
    --lr 0.0001 \
    --warmup-ratio-or-steps 10000 \
    --num-experts $NUM_EXPERTS \
    --k $K \
    --nlayers 4 \
    --nheads 8 \
    --embsize 64 \
    --d-hid 64 \
    --expert-specialization $EXPERT_SPECIALIZATION \
    # --shuffle-buffer-size 0 \
    # --checkpoint-dir "/home/hauke.schuele/save/pretrain-distributed-[257990]-2025-11-18_00-27-20" \