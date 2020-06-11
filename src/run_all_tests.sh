#!/bin/bash

#script="/media/a.rodan/hd-23/dcn/git/Pneumonia-Detection-from-Chest-X-Rays-with-Robustness-to-Deformations/src/train_locally_dcn.py"

script="./train_locally_dcn.py"	#RELATIVE PATHS :( (DOCKER CONSTRAINTS)
epochs=50

#FUCK THIS SHIT
python="/usr/bin/python3.6"

#KAGGLE PNEUMONIA, KAGGLE RSNA, CHEXPERT
dataset="KAGGLE RSNA"


echo "[INFO]	Running test 1: AlexNet last layer, no DCN, $dataset, $epochs epochs, batch size 64"
python $script --log AlexNet_vanilla_DS2 --image AlexNet_vanilla_DS2 --arch AlexNet --epochs $epochs --data_set '../../../KAGGLE_RSNA/data/' &
wait

echo "[INFO]	Running test 2: AlexNet last layer, with DCN, $dataset, $epochs epochs, batch size 64"
python $script --log AlexNet_DCN_DS2 --image AlexNet_DCN_DS2 --arch AlexNet --dcn --epochs $epochs --data_set '../../../KAGGLE_RSNA/data/' &
wait

echo "[INFO]	Running test 3: ResNet18, no DCN, $dataset, $epochs epochs, batch size 64"
python $script --log ResNet18_vanilla_DS2 --image ResNet18_vanilla_DS2 --arch ResNet18 --epochs $epochs --data_set '../../../KAGGLE_RSNA/data/' &
wait

echo "[INFO]	Running test 4: ResNet18, DCN, $dataset, $epochs epochs, batch size 64"
python $script --log ResNet18_DCN_DS2 --image ResNet18_DCN_DS2 --arch ResNet18 --dcn --epochs $epochs --data_set '../../../KAGGLE_RSNA/data/' &
wait

echo "[INFO]	Script finished successfully"

