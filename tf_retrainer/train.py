from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

sys.path.append('/repos/coding-interviews/stackline/')  # point to the stackline repo

from image_classifier import conv_net
from image_reader import load_data


MODEL_PATH = 'models/stackline_inception.pb'



slim = tf.contrib.slim


def build_graph():
    images = load_data()
    print(images)
    predictions = conv_net(images)

    one_hot_labels = slim.one_hot_encoding(1, 63)
    tf.losses.softmax_cross_entropy(one_hot_labels, predictions)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    optimizer = tf.train.AdamOptimizer(0.009)

    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True, )
    slim.learning.train(train_op, LOG_DIR, number_of_steps=10, save_summaries_secs=20)


def main(args):

    build_graph()
    print("HERE")

    # graph = tf.Graph()
    # with graph.as_default():
    #     saver = tf.train.Saver()





if __name__ == "__main__":
    tf.app.run()
