#!/usr/bin/env bash

echo "lets start training..."
if [ ! -d tensorflow/ ]; then
    echo "cloning tensorflow repo..."
    git clone https://github.com/messiest/tensorflow.git  # clone tensorflow from custom build
fi


# download the training images
if [ ! -d /tmp/stackline/training/ ]; then
    echo "downloading the training images..."
    python3 image_download.py training
fi


# retrain the last layer of the inception net on the training data
echo "retraining the model..."
python3 tensorflow/tensorflow/examples/image_retraining/retrain.py \
    --image_dir /tmp/stackline/training/ \
    --output_graph models/stackline.pb\
    --intermediate_output_graphs_dir logs/intermediate_graph/\
    --intermediate_store_frequency 500\
    --how_many_training_steps 5000\
    --output_labels data/predictions/output_labels.txt\
    --summaries_dir logs/retrain_logs\
    --print_misclassified_test_images True\
    --model_dir models/ \
    --learning_rate 0.009 \
    --architecture mobilenet_0.50_224\
    --final_tensor_name stackline-output

tensorboard --logdir=logs/retrain_logs/train/  # launch tensorboard

#if [ ! -d /tmp/stackline/validation/ ]; then
#    echo "downloading the assessment images..."
#    python3 image_download.py validation
#fi
