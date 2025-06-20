# SMP-ppi 
This repository is for the protein-protein interaction task.

## 1. Environment Setup

```bash
git clone https://github.com/Split-and-Merge-Proxy/smp-ppi.git
cd smp-ppi
conda create -n smp-ppi python=3.9
conda activate smp-ppi
pip install -r requirements.txt
```

## 2. Data Preparation
```bash
bash ./scripts/0.prepare-dataset.sh
```


<!-- You can download the ppi data from [SMP - Harvard Dataverse](https://doi.org/10.7910/DVN/22AUPR) and place it in the `./data` folder. -->

## 3. Training (Optional)
### PPITrans
```bash
bash ./scripts/equidock/dist_train.sh
```
**Note:** you can change the `data_fraction` in the Shell file to determine the amount of training data.

### SMP
```bash
bash ./scripts/smp/dist_pretrain.sh
```
**Note:** you can change the `data_fraction` in the Shell file to determine the amount of fine-tuning data (pre-training data does not support yet) and `resume_ckpt` to your own directory.



## 4. Evaluations
### PPITrans
```bash
# Pytorch launcher
bash ./scripts/equidock/dist_inference.sh
bash ./scripts/equidock/dist_eval.sh
```

### SMP
```bash
# Pytorch launcher
bash ./scripts/smp/dist_inference.sh
bash ./scripts/smp/dist_eval.sh
```

## Acknowledges
- [PPITrans](https://github.com/LtECoD/PPITrans)
- [D-SCRIPT](https://github.com/samsledje/D-SCRIPT)
- [ESM](https://github.com/facebookresearch/esm)
- [ProstT5](https://github.com/mheinzinger/ProstT5)
- [fairseq](https://github.com/facebookresearch/fairseq)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)