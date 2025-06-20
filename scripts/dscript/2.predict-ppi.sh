#! /bin/bash

DATASET=dscript
ARCH=ppi

TEST_SET=('human_test')

for set in ${TEST_SET[*]}; do
        # with ppm
        CUDA_VISIBLE_DEVICES=7 python module/predict.py \
        --user-dir module \
        --task ppi \
        \
        --arch ${ARCH} \
        --path  ./save/${DATASET}/restore_${ARCH}/checkpoint_best.pt \
        --emb-dim 1024 \
        --hid-dim 256 \
        \
        --results-path  ./results/${DATASET}/restore_${ARCH}/prediction  \
        --rep-path ./results/${DATASET}/restore_${ARCH}/rep \
        \
        --data-dir ./data/${DATASET}/processed \
        --max-len 800 \
        --gen-subset  ${set}  \
        --batch-size 32

done

