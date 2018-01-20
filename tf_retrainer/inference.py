import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from image_reader import load_data


FROZEN_GRAPH = 'models/stackline.pb'


def load_frozen_graph(pb_file):
    filename = pb_file

    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        images = load_data()

        iterator = images.make_initializable_iterator()
        next_element = iterator.get_next()

        tf.import_graph_def(
            graph_def,
            # usually, during training you use queues, but at inference time use placeholders
            # this turns into "input
            input_map={"input:0": next_element},
            return_elements=None,
            # if input_map is not None, needs a name
            name="bla",
            op_dict=None,
            producer_op_list=None
        )

    checkpoint_path = tf.train.latest_checkpoint("/tmp/checkpoints/")

    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
        saver.restore(sess, checkpoint_path)

        sess.run(iterator.initializer)

        print("NEXT", next_element)

        while True:
            try:
                output = sess.run('stackline-output', feed_dict={'input:0', next_element})
                print("output", output)
            except ValueError:
                continue
            except tf.errors.OutOfRangeError:
                break



def main():
    load_frozen_graph(FROZEN_GRAPH)


if __name__ == "__main__":
    main()
