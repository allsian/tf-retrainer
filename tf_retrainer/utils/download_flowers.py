import os
import sys

import tensorflow as tf

sys.path.append('/repos/tensorflow/models/research/slim')

from datasets import dataset_utils

url = 'http://download.tensorflow.org/data/flowers.tar.gz'
flowers_data_dir = '/tmp/flowers'


def main():
    if not tf.gfile.Exists(flowers_data_dir):
        tf.gfile.MakeDirs(flowers_data_dir)

    dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)


if __name__ == "__main__":
    main()
