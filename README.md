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
**Note:** Please download the pre-trained ProtT5 checkpoint and place it in the `./prot_t5_xl_uniref50` folder. You can also modify the `pair_dir`, `seq_dir`, and `processed_dir` in the shell script to prepare different datasets.

Aditionally, we have already uploaded the processed ppi data, you can directly download it from [SMP - Harvard Dataverse](https://doi.org/10.7910/DVN/0QURCP) and place it in the `./data` folder.



## 3. Training
### PPITrans
```bash
bash ./scripts/1.ppitrans-train.sh
```
**Note:** You can change the `DATASET` in the shell script to determine the dataset (D-SCRIPT or HIPPIE).

### SMP
```bash
bash ./scripts/1.smp-pretrain.sh
bash ./scripts/1.smp-finetune.sh
```
**Note:** You can change the `DATASET` variable in the fine-tuning shell script to select the dataset, and change `finetune-from-model` or `restore-file` to point to your own directory.


## 4. Evaluations
```bash
bash ./scripts/test.sh
```
**Note:** You can set `DATASET` and `TEST_SET` to either D-SCRIPT or HIPPIE, and change `path` to point to your own directory containing the trained checkpoint.

## 5. Reproducing the Results Reported in the Manuscript
To reproduce the results reported in our manuscript, first download the processed test sets (`dscript_test.zip` for the D-SCRIPT dataset and `hippie.zip` for the HIPPIE dataset) from https://doi.org/10.7910/DVN/0QURCP and unzip them. We have also uploaded the trained checkpoints (both PPITrans and SMP) in the `./ckpts` folder; please download them and place them in your desired directory.

Next, set the `DATASET` and `TEST_SET` to specify which dataset to evaluate, update `path` to point to your directory containing the trained checkpoint, and then run the following command:

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


## 6. Inference on Your Custom Data
We have already uploaded the trained weights of SMP in the `./ckpts`, you can directly download it and place it in your own directory. Additionally, we provide an example in `./example` that demonstrates how to perform inference on your own custom data. You can run it with:

```bash
python -u custom_inference.py
```
The output will print whether the two input proteins interact.

## Acknowledges
- [PPITrans](https://github.com/LtECoD/PPITrans)
- [D-SCRIPT](https://github.com/samsledje/D-SCRIPT)
- [ESM](https://github.com/facebookresearch/esm)
- [ProtTrans](https://github.com/agemagician/ProtTrans)
- [fairseq](https://github.com/facebookresearch/fairseq)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)