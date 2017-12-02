# ECE544 NA FA17 project Distracted Driver Action Recognition
# vgg 16 implementation
# author: Fangwen Wu

import tensorflow as tf

from utils import data_tools
from models.vgg16 import Vgg16
from train_eval_model import train_model, eval_model


def main(_):
    image_size = 224
    learning_rate = 0.01
    batch_size = 32
    training_epoch = 1
    keep_prob = 0.5
    # Get training dataset and Data processing
    #x_train, y_train = data_tools.read_and_normalize_train_data(
    #    image_size, image_size)

    # Build model
    model = Vgg16(image_size, image_size)

    # Start training
    model = train_model(model,learning_rate, batch_size,
                        training_epoch, keep_prob)

    # Start testing
    #x_test, x_test_id = data_tools.read_and_normalize_test_data(
    #    image_size, image_size)
    #eval_model(model, x_test, x_test_id)


if __name__ == "__main__":
    tf.app.run()
