#! /bin/bash

set -x

conda activate ppi
module load cuda/11.8
module load cudnn/8.7.0_cu11x
module load gcc/9.3.0

DATASET=hippie
ARCH=ppi
CRITEION=ppi_crossentropy


NODES=$1
JOB_NAME=$2
GPUS=$3
GPUS_PER_NODE=$4
CPUS_PER_TASK=$5


srun -p vip_gpu_ailab \
     -A ai4astro \
    --nodes=${NODES} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    fairseq-train \
        --user-dir module \
        --save-dir ./save/${DATASET}_slurm/${ARCH} \
        --seed 100 \
        \
        --optimizer adam \
        --lr 3e-5 \
        --batch-size 32 \
        --max-epoch 5 \
        \
        --data-dir ./data/${DATASET}/processed \
        --train-subset hippie_train \
        --valid-subset hippie_val \
        --max-len 800 \
        \
        --task ppi \
        --arch ${ARCH} \
        --criterion ${CRITEION} \
        \
        --dropout 0.2 \
        --emb-dim 1024 \
        --hid-dim 256 \
        --trans-layers 8 \
