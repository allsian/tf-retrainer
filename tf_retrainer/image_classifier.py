import sys
import tensorflow as tf
sys.path.append('/repos/coding-interviews/stackline')
# from utils import preprocess


slim = tf.contrib.slim


def conv_net(images, n_classes=None):
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(images, 20, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='maxpool1')
        net = slim.conv2d(net, 50, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='maxpool2')
        net = slim.flatten(net, scope='flatten3')
        net = slim.fully_connected(net, 500, scope='dense4')
        out = slim.fully_connected(net, None, activation_fn=None, scope='dense')

    return out
