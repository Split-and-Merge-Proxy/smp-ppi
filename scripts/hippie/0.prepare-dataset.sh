#! /bin/bash
python data/hippie/builddataset.py \
    --pair_dir  ./data/hippie/data/pairs   \
    --seq_dir   ./data/hippie/data/seqs    \
    --processed_dir ./data/hippie/processed    \
    --max_length    800 \
    --min_length    20  

python data/hippie/embed.py  \
    --pretrained_model ./prot_t5_xl_uniref50 \
    --processed_dir ./data/hippie/processed \
    --device 7 \
    --batch_size 32