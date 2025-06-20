#! /bin/bash

set -x

DATASET=hippie
ARCH=ppi
CRITEION=ppi_crossentropy


# with ppm
CUDA_VISIBLE_DEVICES=7 fairseq-train \
    --user-dir module \
    --save-dir ./save/${DATASET}/finetune_${ARCH}_best \
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
    --finetune-from-model ./save/${DATASET}/pseudo_${ARCH}/checkpoint_best.pt 
