import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.framework import ops, dtypes

DATA_DIR = '/tmp/stackline/{}/'


def _parse_function(filename):
    print(filename)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.resize_image_with_crop_or_pad(image_decoded, 224, 224)  # to match mobilenet size
    image_resized = tf.to_float(image_decoded)

    return image_resized


def load_data(dataset='validation'):
    global DATA_DIR
    DATA_DIR = DATA_DIR.format(dataset)
    data = pd.read_csv('data/processed/{}.csv'.format(dataset))

    data = data[data['FileType'] != 'gif']  # no gifs!

    all_files = list(data['FilePath'])
    all_files = all_files[:20]

    images = ops.convert_to_tensor(all_files, dtype=dtypes.string)

    image_data = tf.data.Dataset.from_tensor_slices(images)
    image_data = image_data.map(_parse_function)

    return image_data


if __name__ == "__main__":
    load_data()
