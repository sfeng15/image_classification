"""Input and output helpers to load in data.
"""
import os
import cv2
import pickle
import glob


def read_image(path, rows, cols):
    img = cv2.imread(path)
    resized = cv2.resize(img, (rows, cols))
    return resized


def load_train(data_txt_file, image_data_path, rows, cols):

    image_train = []
    label_train = []
    print('Read drivers data')

    # get driver data
    f = open(data_txt_file, 'r')
    lines = f.readlines()[1:]
    f.close()

    print('Read train image')
    for line in lines:
        line = line.strip()
        path = image_data_path + '/' + \
            str(line.split(',')[1]) + '/' + str(line.split(',')[2])
        image_train.append(read_image(path, rows, cols))
        label_train.append(int(line.split(',')[1][1]))
    return image_train, label_train


def load_test(rows, cols):
    print('Read test images')
    path = os.path.join('..', 'data', 'test', '*.jpg')
    files = glob.glob(path)
    image_test = []

    for f in files:
        image_test.append(read_image(f, rows, cols))

    return image_test


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
