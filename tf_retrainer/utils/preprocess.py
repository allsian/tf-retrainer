from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf


slim = tf.contrib.slim


def image(image, height, width):
    """
    resize and normalize image

    :param image: a tensor representing an image
    :type image: tensorflow tensor
    :param height: pixel height of output image
    :type height: int
    :param width: pixel width of output image
    :type width: int
    :return: processed image tensor
    :rtype: tensorflow tensor
    """
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(image, width, height)
    image = tf.subtract(image, 128)
    image = tf.div(image, 128)

    return image


if __name__ == "__main__":
    img = ndimage.imread('data/dev_test/me.jpg')
    image_tensor = tf.convert_to_tensor(img)

    image_tensor = image(image_tensor, 28, 28)

    print(image_tensor)

    with tf.Session() as sess:
        # display encoded back to image data
        image = sess.run(image_tensor)

    print(type(image), image)
    plt.imshow(image)
    plt.show()
