#! /bin/bash

DATASET=dscript
ARCH=ppi
CRITEION=ppi_crossentropy

TEST_SET=('human_test')

for split in ${TEST_SET[*]}; do

    CUDA_VISIBLE_DEVICES=7 python module/evaluate.py \
        --result_dir ./results/${DATASET}/finetune_${ARCH}_best/prediction  \
        --metric_dir ./results/${DATASET}/finetune_${ARCH}_best/metric \
        --split ${split}
done