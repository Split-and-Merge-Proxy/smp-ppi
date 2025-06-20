#! /bin/bash

DATASET=hippie
ARCH=ppi

TEST_SET=('hippie_test')

for set in ${TEST_SET[*]}; do
        # with ppm
        CUDA_VISIBLE_DEVICES=7 python module/predict.py \
        --user-dir module \
        --task ppi \
        \
        --arch ${ARCH} \
        --path  ./save/${DATASET}/restore_${ARCH}_best/checkpoint_best.pt \
        --emb-dim 1024 \
        --hid-dim 256 \
        \
        --results-path  ./results/${DATASET}/restore_${ARCH}_best/prediction  \
        --rep-path ./results/${DATASET}/restore_${ARCH}_best/rep \
        \
        --data-dir ./data/${DATASET}/processed \
        --max-len 800 \
        --gen-subset  ${set}  \
        --batch-size 32

done

