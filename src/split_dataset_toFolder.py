import sys
import split_folders


if __name__ == '__main__':
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    data_dir = '../../../chest_xray_pneumonia/chest_xray'
    split_folders.ratio(data_dir, output="output", seed=1337, ratio=(.6, .2, .2)) # default