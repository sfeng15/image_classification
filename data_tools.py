import os
import numpy as np
import tensorflow as tf
import utils.io_tools as io_tools

def read_and_normalize_train_data(rows, cols):


    print('Read train image')
    for line in lines:
        line = line.strip()
        path = image_data_path + '/' + \
            str(line.split(',')[1]) + '/' + str(line.split(',')[2])
        image_train.append(read_image(path, rows, cols))
        label_train.append(int(line.split(',')[1][1]))
        
    train_image, train_label = io_tools.load_train(
            'data/driver_imgs_list.csv', 'data/train', rows, cols)


    train_image = np.array(train_image, dtype=np.uint8)
    train_label = np.array(train_label, dtype=np.uint8)

    train_label = tf.keras.utils.to_categorical(train_label, 10)
    train_image = train_image.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]

    for c in range(3):
        train_image[:, :, :, c] = train_image[:, :, :, c] - mean_pixel[c]

    print('Train shape:', train_image.shape)
    print(train_image.shape[0], 'train samples')

    return train_image, train_label


def read_and_normalize_test_data(rows, cols):
    cache_path = 'data/cache/test_data.dat'
    if not os.path.isfile(cache_path):
        (test_image, test_image_id) = io_tools.load_test(rows, cols)
        io_tools.cache_data((test_image, test_image_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_image, test_image_id) = io_tools.restore_data(cache_path)

    test_image = np.array(test_image, dtype=np.uint8)

    test_image = test_image.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_image[:, :, :, c] = test_image[:, :, :, c] - mean_pixel[c]

    print('Test shape:', test_image.shape)
    print(test_image.shape[0], 'test samples')
    return test_image, test_image_id
