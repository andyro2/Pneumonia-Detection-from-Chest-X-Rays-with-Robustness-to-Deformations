import pydicom as dicom
import os
import cv2

folder_path = '../../../KAGGLE_RSNA/healthy'
jpg_folder_path = '../../../KAGGLE_RSNA/healthy_out'

# make it True if you want in PNG format
PNG = False

images_path = os.listdir(folder_path)
for n, image in enumerate(images_path):
    ds = dicom.dcmread(os.path.join(folder_path, image))
    pixel_array_numpy = ds.pixel_array
    if PNG == False:
        image = image.replace('.dcm', '.jpg')
    else:
        image = image.replace('.dcm', '.png')
    cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
    if n % 50 == 0:
        print('{} image converted'.format(n))