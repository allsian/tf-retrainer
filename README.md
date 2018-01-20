# Stackline: Data Sciene Coding Challenge
### [Christopher Messier](https://messiest.github.io)
- Coding Interview date/time: 1/20/2018 8am EST
- Available to start: 4/1/2018 (I'm currently teaching a course that ends on 3/22/2018)
- Salary expectation: $85 - $90k

# Documentation

### Running the program

Navigate to this folder in your terminal, and enter the following command:
```bash
bash ./demo.sh
```
This will download the images to disk, prep the images, and train a model on the provided dataset.

### Libraries Used
- [pandas]()
- [explor]()
- [scikit-learn]()
- [tensorflow]()
- [tensorboard]()
- [nltk]()

## Overview

You will also have to update some file paths to point to the right locations on the disk.
These can be found
- `run.py`
- `utils/download_flowers.py`
- `utils/preprocess.py`
- `models/image_classifier.py`

## Data Preparation

### Data Summary

- File Type?
- Dataset Size?

### Data Processing and Exploratory Data Analysis (EDA)


### Pre-processing

## Model

The model that I developed for this task was built with the [TensorFlow](https://www.tensorflow.org/) framework.
To build the model I utilized the `tf.slim`, a high-level wrapper for TensorFlow.
In a production environment I would prefer to work directly with TensorFlow, but with the time constraints that come with this task, `tf.slim` makes the development process fast.
Using `tf.slim` simplifies the design and implementation of deep learning models by providing a high-level interface to TensorFlow.
You can read more about it [here]().


### Model Architecture

The model is a pretrained model


### Training

In order to take advantage of GPU parallelization, I trained the model on an Amazon Web Services (AWS) Elastic Cloud Computing (EC2) p2.xlarge instance, running the Amazon Deep Learning.
This instance comes with Nvidia Tesla K80 GPUs, which accelerate the training time considerably.
The AMI that I'm using also simplifies the training process.
It uses Amazon Linux as the operating system, and not only comes with the source code for the most popular deep learning libraries, (TensorFlow, Theano, PyTorch, etc.), but is also configured with CUDA 8.
This is essential, because at the time of this work, TensorFlow does not support the most recent builds of CUDA.

## Evaluation
