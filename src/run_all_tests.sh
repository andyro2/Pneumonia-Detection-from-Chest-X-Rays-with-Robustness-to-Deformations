#!/bin/bash

#script= "/media/a.rodan/hd-23/dcn/git/Pneumonia-Detection-from-Chest-X-Rays-with-Robustness-to-Deformations/src/train_locally_dcn.py"
script="./train_locally_dcn.py"
epochs=1
python="/usr/bin/python3.6"

echo "Running test 1: AlexNet last layer, no DCN, Kaggle Pneumonia, 25 epochs, batch size 64"
python $script --log AlexNet_vanilla_DS1 --image AlexNet_vanilla_DS1 --epochs $epochs &
wait

echo "Running test 2: AlexNet last layer, with DCN, Kaggle Pneumonia, 25 epochs, batch size 64"
python $script --log AlexNet_DCN_DS1 --image AlexNet_DCN_DS1 --dcn --epochs $epochs &
wait

echo "Running test 3: ResNet18, no DCN, Kaggle Pneumonia, 25 epochs, batch size 64"
python $script --log ResNet18_vanilla_DS1 --image ResNet18_vanilla_DS1 --epochs $epochs &
wait

echo "Running test 1: ResNet18, DCN, Kaggle Pneumonia, 25 epochs, batch size 64"
python $script --log ResNet18_DCN_DS1 --image ResNet18_DCN_DS1 --epochs $epochs &
wait

echo "Script finished successfully"

