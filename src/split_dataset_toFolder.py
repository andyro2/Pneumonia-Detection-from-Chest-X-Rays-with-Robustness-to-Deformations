import sys
import split_folders


if __name__ == '__main__':
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    data_dir = '../../Kaggle_RSNA/data'
    split_folders.ratio(data_dir, output="output", seed=1337, ratio=(.6, .2, .2)) # default



###################################################################################################
# import pandas as pd
# import os
# path = '../../../rsna-pneumonia-detection-challenge/'
# data_dir = 'data/'
# train_dir =  path + 'train'
# test_dir =  path + 'test'
# val_dir =  path + 'val'
#
# labels = pd.read_csv(path + 'stage_2_train_labels.csv',usecols = ['patientID','Target'])
# print(labels)
# # create validation directory
# if not os.path.exists(train_dir):
#     os.mkdir(train_dir)
#     os.mkdir(test_dir)
#     os.mkdir(val_dir)
# # .. and class directories
#
#
#
# n = 0.2
# # move all pics in class folders
#
# for p in labels.itertuples():
#     filepath = data_dir + 'train/{p.id}.jpg'
#     trainpath = data_dir + 'train/{p.breed}/{p.id}.jpg'
#
# # os.rename(f'{file_path}', f'{train_path}')
#
# # move valid pics to valid folder
#
# for f in os.listdir(data_dir + 'train'):
#     pics = os.listdir(data_dir + 'train/{f}')
#     numpics = len(pics)
#     numvalpics = round(n * numpics)
#
# # val_pics = random.sample(pics,  numvalpics)
#
# for p in val_pics:
#     file_path = data_dir + 'train/{f}/{p}'
#     val_path = data_dir + 'valid/{f}/{p}'
#
#     # os.rename(f'{file_path}', f'{val_path}')
