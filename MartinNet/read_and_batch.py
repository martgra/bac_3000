import tensorflow as tf
import skimage.io as io

IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299

tfrecords_filename ='/home/jason/test/segmentation.tfrecords'
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=1)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example, features={
                                        'height': tf.FixedLenFeature([], tf.int64),
                                        'width': tf.FixedLenFeature([], tf.int64),
                                        'image_raw': tf.FixedLenFeature([], tf.string),
                                        'label': tf.FixedLenFeature([], tf.int64)
                                        }
    )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([height, width, 3])

    image = tf.reshape(image, image_shape)
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)

    resized_image = tf.image.resize_image_with_crop_or_pad(
                                            image=image,
                                            target_height=IMAGE_HEIGHT,
                                            target_width=IMAGE_WIDTH)
    images, labels = tf.train.shuffle_batch([resized_image, label],
                                            batch_size=32,
                                            capacity=30,
                                            num_threads=2,
                                            min_after_dequeue=10)

    return images, labels

x, y = read_and_decode(filename_queue)
print(x,y)
