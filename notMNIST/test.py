import tensorflow as tf
import os


def test(folder):
    image_files = os.listdir(folder)
    ar = tf.image.decode_jpeg(image_files[2])
    return ar
