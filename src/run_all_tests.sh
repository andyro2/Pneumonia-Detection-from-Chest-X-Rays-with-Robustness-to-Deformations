#!/bin/bash

#script="/media/a.rodan/hd-23/dcn/git/Pneumonia-Detection-from-Chest-X-Rays-with-Robustness-to-Deformations/src/train_locally_dcn.py"

#FUCK THIS SHIT
python="/usr/bin/python3.6"
script="./train_locally_dcn.py"	#RELATIVE PATHS :( (DOCKER CONSTRAINTS)

epochs=25
batch_size=64

#KAGGLE PNEUMONIA, KAGGLE RSNA, CHEXPERT
#dataset="Kaggle_RSNA" #1
#dataset='../../../KAGGLE_RSNA/data/' #2
dataset='../../kaggle_small/' # small





echo [INFO] TEST1:	RESNET50 VANILLA	DS1
echo [INFO] TEST2:	RESNET50 DCN		DS1
echo [INFO] TEST3:	RESNET50 VANILLA	DS2
echo [INFO] TEST4:	RESNET50 DCN		DS2


echo "[INFO] Running test 1: ResNet50, DCN=FALSE, DS1, EPOCHS: $epochs, BATCH SIZE: $batch_size"
python $script --log ResNet50_vanilla_ds1 --image ResNet50_vanilla_ds1 --arch ResNet50 --epochs $epochs  &
wait

echo "[INFO] Running test 1: ResNet50, DCN=TRUE, DS1, EPOCHS: $epochs, BATCH SIZE: $batch_size"
python $script --dcn --log ResNet50_dcn_ds1 --image ResNet50_dcn_ds1 --arch ResNet50 --epochs $epochs  &
wait

echo "[INFO] Running test 1: ResNet50, DCN=FALSE, DS2, EPOCHS: $epochs, BATCH SIZE: $batch_size"
python $script --log ResNet50_vanilla_ds2 --image ResNet50_vanilla_ds2 --arch ResNet50 --epochs $epochs --data_set '../../../KAGGLE_RSNA/data/' &
wait

echo "[INFO] Running test 1: ResNet50, DCN=TRUE, DS2, EPOCHS: $epochs, BATCH SIZE: $batch_size"
python $script --dcn --log ResNet50_dcn_ds2 --image ResNet50_dcn_ds2 --arch ResNet50 --epochs $epochs --data_set '../../../KAGGLE_RSNA/data/' &
wait

echo "[INFO] Script finished successfully"


echo "[INFO] Running test 1: ResNet50, DCN=TRUE, DS2, EPOCHS: $epochs, BATCH SIZE: $batch_size"
python $script --dcn --log simple_cnn_shortDS --image simple_cnn_shortDS --arch simple_cnn --epochs $epochs --data_set '../../kaggle_small/' &
wait
