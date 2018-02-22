#!/usr/bin/env bash

cd tf_retrainer

if [ ! -f cifar-extender/cifar_extender/data ]; then
    bash cifar-extender/demo.sh
fi

echo "retraining the model..."
# retrain the last layer of the inception net on the training data
python3 image_retraining/retrain.py \
    --image_dir cifar-extender/cifar_extender/data/images/\
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
