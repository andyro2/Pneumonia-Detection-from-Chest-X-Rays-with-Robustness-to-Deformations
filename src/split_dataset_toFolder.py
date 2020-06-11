import os
import random

if __name__ == '__main__':
    # SET DIRECTORIES
    # healthy_dir   = '/media/a.rodan/hd-23/dcn/KAGGLE_RSNA/data/healthy'
    healthy_dir   = '../../../KAGGLE_RSNA/data/healthy'
    sick_dir      = '../../../KAGGLE_RSNA/data/sick'
    healthy_train = '../../../KAGGLE_RSNA/data/healthy_train'
    healthy_val   = '../../../KAGGLE_RSNA/data/healthy_val'
    healthy_test  = '../../../KAGGLE_RSNA/data/healthy_test'
    sick_train    = '../../../KAGGLE_RSNA/data/sick_train'
    sick_val      = '../../../KAGGLE_RSNA/data/sick_val'
    sick_test     = '../../../KAGGLE_RSNA/data/sick_test'

    # SET NUMBERS
    list_healthy   = os.listdir(healthy_dir)
    list_sick      = os.listdir(sick_dir)
    number_healthy = len(list_healthy)
    number_sick    = len(list_sick)

    # 60-20-20
    number_healthy_train = int(number_healthy * 0.6)
    number_healthy_val   = int(number_healthy * 0.2)
    number_healthy_test  = number_healthy - (number_healthy_train + number_healthy_val)
    number_sick_train    = int(number_sick * 0.6)
    number_sick_val      = int(number_sick * 0.2)
    number_sick_test     = number_sick - (number_sick_train + number_sick_val)

    # PRINT INFO
    print('-I- Splitting data into TRAIN, VALIDATION, TEST groups')
    print('-I- NUMBER OF HEALTHY IMAGES:            ', number_healthy)
    print('-I- NUMBER OF HEALTHY TRAIN IMAGES:      ', number_healthy_train)
    print('-I- NUMBER OF HEALTHY VALIDATION IMAGES: ', number_healthy_val)
    print('-I- NUMBER OF HEALTHY TEST IMAGES:       ', number_healthy_test)
    print('-I- NUMBER OF SICK IMAGES:               ', number_sick)
    print('-I- NUMBER OF SICK TRAIN IMAGES:         ', number_sick_train)
    print('-I- NUMBER OF SICK VALIDATION IMAGES:    ', number_sick_val)
    print('-I- NUMBER OF SICK TEST IMAGES:          ', number_sick_test)

    ##   SPLIT HEALTHY
    ##   TRAIN
    #counter = 0
    #while counter < number_healthy_train:
    #    random_file = random.choice(list_healthy)
    #    if random_file == '.':
    #        continue
    #    if random_file == '..':
    #        continue
    #    counter = counter + 1
    #    try:
    #        os.rename(healthy_dir + '/' + random_file, healthy_train + '/' + random_file)
    #        print('-I- RANDOM FILE', counter, 'IS:                    ', random_file)
    #    except:
    #        print('-E- RANDOM FILE', counter, 'FAILED TO MOVE (TRAIN):', random_file)
#
#
    ##   VAL
    #counter = 0
    #while counter < number_healthy_val:
    #    random_file = random.choice(list_healthy)
    #    if random_file == '.':
    #        continue
    #    if random_file == '..':
    #        continue
    #    counter = counter + 1
    #    try:
    #        os.rename(healthy_dir + '/' + random_file, healthy_val + '/' + random_file)
    #        print('-I- RANDOM FILE', counter, 'IS:                    ', random_file)
    #    except:
    #        print('-E- RANDOM FILE', counter, 'FAILED TO MOVE (TEST): ', random_file)
#
    ##   TEST
    #counter = 0
    #while counter < number_healthy_test:
    #    random_file = random.choice(list_healthy)
    #    if random_file == '.':
    #        continue
    #    if random_file == '..':
    #        continue
    #    counter = counter + 1
    #    try:
    #        os.rename(healthy_dir + '/' + random_file, healthy_test + '/' + random_file)
    #        print('-I- RANDOM FILE', counter, 'IS:                    ', random_file)
    #    except:
    #        print('-E- RANDOM FILE', counter, 'FAILED TO MOVE (VAL):  ', random_file)
#


    #   SPLIT SICK
    #   TRAIN
    counter = 0
    while counter < number_sick_train:
        random_file = random.choice(list_sick)
        if random_file == '.':
            continue
        if random_file == '..':
            continue
        counter = counter + 1
        try:
            os.rename(sick_dir + '/' + random_file, sick_train + '/' + random_file)
            print('-I- RANDOM FILE', counter, 'IS:                    ', random_file)
        except:
            print('-E- RANDOM FILE', counter, 'FAILED TO MOVE (TRAIN):', random_file)


    #   VAL
    counter = 0
    while counter < number_sick_val:
        random_file = random.choice(list_sick)
        if random_file == '.':
            continue
        if random_file == '..':
            continue
        counter = counter + 1
        try:
            os.rename(sick_dir + '/' + random_file, sick_val + '/' + random_file)
            print('-I- RANDOM FILE', counter, 'IS:                    ', random_file)
        except:
            print('-E- RANDOM FILE', counter, 'FAILED TO MOVE (TEST): ', random_file)

    #   TEST
    counter = 0
    while counter < number_sick_test:
        random_file = random.choice(list_sick)
        if random_file == '.':
            continue
        if random_file == '..':
            continue
        counter = counter + 1
        try:
            os.rename(sick_dir + '/' + random_file, sick_test + '/' + random_file)
            print('-I- RANDOM FILE', counter, 'IS:                    ', random_file)
        except:
            print('-E- RANDOM FILE', counter, 'FAILED TO MOVE (VAL):  ', random_file)


    print('-I- FINISHED')


#import split_folders


#if __name__ == '__main__':
#    # Split with a ratio.
#    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#    data_dir = '../../../KAGGLE_RSNA/data/'
#    split_folders.ratio(data_dir, output="output", seed=1337, ratio=(.6, .2, .2)) # default



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
