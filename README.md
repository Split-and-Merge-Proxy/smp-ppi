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
**Note:** you should download the pre-trained ProtT5 ckpt and put it in the `./prot_t5_xl_uniref50` folder. Additionally, you can change the `pair_dir`, `seq_dir`, and `processed_dir` in the Shell file to prepare different dataset.

You can directly download the ppi data from [SMP - Harvard Dataverse](https://doi.org/10.7910/DVN/0QURCP) and place it in the `./data` folder.



## 3. Training
### PPITrans
```bash
bash ./scripts/ppitrans/1.train-ppi-trans.sh
```
**Note:** you can change the `DATASET` in the Shell file to determine the dataset.

### SMP
```bash
bash ./scripts/smp/1.pre-train-ppi-trans.sh
bash ./scripts/smp/2.fine-tune-ppi-trans.sh
```
**Note:** you can change the `DATASET` in the fine-tune Shell file to determine the dataset and `finetune-from-model` to your own directory.


## 4. Evaluations
### PPITrans
```bash
bash ./scripts/ppitrans/2.predict-ppi.sh
bash ./scripts/ppitrans/3.evaluate-ppi.sh
```

### SMP
```bash
bash ./scripts/smp/3.predict-ppi.sh 
bash ./scripts/smp/4.evaluate-ppi.sh
```

## Acknowledges
- [PPITrans](https://github.com/LtECoD/PPITrans)
- [D-SCRIPT](https://github.com/samsledje/D-SCRIPT)
- [ESM](https://github.com/facebookresearch/esm)
- [ProtTrans](https://github.com/agemagician/ProtTrans)
- [fairseq](https://github.com/facebookresearch/fairseq)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)