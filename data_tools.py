import numpy as np
import tensorflow as tf
import os
import io_tools
from numpy.random import permutation


def train_data_preprocessing(rows, cols):
    cache_path = 'data/cache/train_data.dat'

    if not os.path.isfile(cache_path):
        train_image, train_label = io_tools.load_train(
            'data/driver_imgs_list.csv', 'data/train', rows, cols)
        io_tools.cache_data((train_image, train_label), cache_path)
    else:
        print('Restore training data from cache')
        (train_image, train_label) = io_tools.restore_data(cache_path)

    train_image = np.array(train_image, dtype=np.uint8)
    train_label = np.array(train_label, dtype=np.uint8)

    train_image = train_image.transpose((0, 3, 1, 2))
    train_label = tf.keras.utils.to_categorical(train_label, 10)
    train_image = train_image.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]

    for c in range(3):
        train_image[:, c, :, :] = train_image[:, c, :, :] - mean_pixel[c]

    perm = permutation(len(train_label))
    train_image = train_image[perm]
    train_label = train_label[perm]
    print('Train shape:', train_image.shape)
    print(train_image.shape[0], 'train samples')

    return train_image, train_label


def read_and_normalize_test_data(rows, cols):
    cache_path = 'data/cache/test_data.dat'
    if not os.path.isfile(cache_path):
        test_image = io_tools.load_test(rows, cols)
        io_tools.cache_data(test_image, cache_path)
    else:
        print('Restore test from cache!')
        test_image = io_tools.restore_data(cache_path)

    test_image = np.array(test_image, dtype=np.uint8)

    test_image = test_image.transpose((0, 3, 1, 2))

    test_image = test_image.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_image[:, c, :, :] = test_image[:, c, :, :] - mean_pixel[c]

    print('Test shape:', test_image.shape)
    print(test_image.shape[0], 'test samples')
    return test_image
