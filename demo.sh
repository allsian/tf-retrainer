#!/usr/bin/env bash

echo running the retrainer on CIFAR10 images

cd tf_retrainer  # enter the package directory

pip3 install -r REQUIREMENTS.txt  # install/update required packages

python3 image_download.py 10 1