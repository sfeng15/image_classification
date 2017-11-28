from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import genfromtxt
import skimage
import os
from skimage import io
import csv
import time
import cv2
import pickle
import tensorflow as tf

from utils import io_tools
from utils import data_tools
from models.vgg16 import vgg16
from train_eval_model import train_model, eval_model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epoch', 100, 'Number of epoch to run.')
flags.DEFINE_string('model_type', 'vgg16', 'vgg16 or XX')


def main(_):
    # Get trainingdataset
    io_tools.load_train()

    # Data processing
    data_tools.train_data_preprocessing

    # Build model
    model = vgg16()

    # Start training
    model = train_model()

    # Start testing
    io_tools.load_test()
    data_tools.test_data_preprocessing
    eval_model()

if __name__ == "__main__":
    tf.app.run()