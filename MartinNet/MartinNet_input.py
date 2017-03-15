import tensorflow as tf
import numpy as np
from PIL import Image
import os
image_files = os.listdir('/home/jason/train/train/Agaricus_bisporus')
labels = []


def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


for i in range(len(image_files)):
    image_files[i] = os.path.join('/home/jason/train_3/train/Agaricus_bisporus', image_files[i])
    labels.append(0)

tffiles =['/home/jason/test.tfrecords']

filename_queue = tf.train.string_input_producer(image_files)
filename_queue2 = tf.train.string_input_producer(tffiles)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value)
init_op = tf.initialize_all_variables()

print(key)

with tf.Session() as sess:
    sess.run(init_op)

# Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    writer = tf.python_io.TFRecordWriter('/home/jason/test.tfrecords')
    for i in range(1):
        image = my_img.eval(session=sess)
        img = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(299),
            'width': _int64_feature(299),
            'depth': _int64_feature(3),
            'label': _int64_feature(0),
            'image_raw': _bytes_feature(img)
        }))
        writer.write(example.SerializeToString())
    writer.close()

    reader = tf.TFRecordReader()
    _, ex = reader.read(filename_queue2)
    features = tf.parse_single_example(ex, features=
    {
        'label': tf.VarLenFeature(tf.int64),
        'image_raw': tf.FixedLenFeature([299, 299, 3], tf.string)
    })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    print(image)
    coord.request_stop()
    coord.join(threads)


