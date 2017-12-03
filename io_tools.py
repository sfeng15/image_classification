"""Input and output helpers to load in data.
"""
import os
import pickle
import glob
import cv2

def read_image(path, rows=224, cols=224):
    img = cv2.imread(path)
    resized = cv2.resize(img, (rows, cols))
    return resized


def load_train(data_txt_file, image_data_path):

    print('Read drivers data')

    # get driver data
    f = open(data_txt_file, 'r')
    lines = f.readlines()[1:]
    f.close()

    return lines


def load_test():
    print('Read test images')
    path = os.path.join('.', 'data', 'test', '*.jpg')
    files = glob.glob(path)

    return files

def cache_data(data, path):
    if not os.path.isdir('data/cache'):
        os.mkdir('data/cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data
