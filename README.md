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

## 5. Reproducing the Results Reported in the Manuscript
To reproduce the results reported in our manuscript, first download the processed test set (`dscript_test.zip`for the D-SCRIPT dataset and `hippie.zip` for the HIPPIE dataset) from https://doi.org/10.7910/DVN/0QURCP and unzip them.

Then, change the `method_name` in the `test.sh` to determine whether eval the EquiDock or SMP method. And we have already provided the ckpts for EquiDock and SMP in the `./checkpts` folder, you can download them and change the `ckpt_path` in the `test.sh` to your own local path, and run the following command:

```bash
bash ./scripts/test.sh
```

The expected results are shown below.

**D-SCRIPT Test Set**

|Method| Recall | F1-Score | AUPR |
|----------|----------|----------|----------|
|PPITrans| 0.487 | 0.640  |  0.775 |
|SMP|  0.594 | 0.708  | 0.788  |


**HIPPIE Test Set**

|Method| Recall |F1-Score | AUPR |
|----------|----------|----------|----------|
|PPITrans| 0.692 | 0.674  | 0.712  |
|SMP| 0.746  | 0.693  | 0.726  |


## 6. Inference on your custom data



## Acknowledges
- [PPITrans](https://github.com/LtECoD/PPITrans)
- [D-SCRIPT](https://github.com/samsledje/D-SCRIPT)
- [ESM](https://github.com/facebookresearch/esm)
- [ProtTrans](https://github.com/agemagician/ProtTrans)
- [fairseq](https://github.com/facebookresearch/fairseq)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)