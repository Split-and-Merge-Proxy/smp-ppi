#! /bin/bash

set -x

DATASET=dscript
ARCH=ppi
CRITEION=ppi_crossentropy


# with ppm
CUDA_VISIBLE_DEVICES=7 fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/pseudo_${ARCH} \
    --seed 100 \
    \
    --optimizer adam \
    --lr 3e-5 \
    --batch-size 32 \
    --max-epoch 10 \
    \
    --data-dir ./data/${DATASET}/processed \
    --train-subset pseudomultimer_train \
    --valid-subset human_test \
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
