#!/bin/bash

#script="/media/a.rodan/hd-23/dcn/git/Pneumonia-Detection-from-Chest-X-Rays-with-Robustness-to-Deformations/src/train_locally_dcn.py"

#FUCK THIS SHIT
python="/usr/bin/python3.6"
script="./train_locally_dcn.py"	#RELATIVE PATHS :( (DOCKER CONSTRAINTS)

epochs=25
batch_size=20

#KAGGLE PNEUMONIA, KAGGLE RSNA, CHEXPERT
dataset="Kaggle_RSNA" #1
#dataset='../../../KAGGLE_RSNA/data/' #2
#dataset='../../kaggle_small/' # small


echo "[INFO] Script finished successfully"


echo "[INFO] Running test 1: simple_cnn_DS1, DCN=False, DS2, EPOCHS: $epochs, BATCH SIZE: $batch_size"
python $script --log simple_cnn_DS1 --image simple_cnn_DS1 --arch simple_cnn --epochs $epochs --batch_size $batch_size &
wait

echo "[INFO] Running test 2: ResNet50, DCN=TRUE, DS2, EPOCHS: $epochs, BATCH SIZE: $batch_size"
python $script --dcn --log simple_cnn_dcn_DS1 --image simple_cnn_dcn_DS1 --arch simple_cnn --epochs $epochs --batch_size $batch_size &
wait
