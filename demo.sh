#!/usr/bin/env bash

echo "retraining the model..."
# retrain the last layer of the inception net on the training data
python3 tf_retrainer/image_retraining/retrain.py \
    --image_dir cifar-extender/images/\
    --output_graph models/stackline.pb\
    --intermediate_output_graphs_dir logs/intermediate_graph/\
    --intermediate_store_frequency 100\
    --how_many_training_steps 500\
    --output_labels data/predictions/output_labels.txt\
    --summaries_dir logs/retrain_logs\
    --print_misclassified_test_images True\
    --model_dir models/ \
    --learning_rate 0.009 \
    --architecture inception_v3\
    --final_tensor_name stackline-output
