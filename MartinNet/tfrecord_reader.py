import matplotlib
import numpy as np
import skimage.io as io
from PIL import Image
import tensorflow as tf
import os
'''
kode for å demonstrere lesing og lagring av bilder med np.
root = '/home/jason/test/img/'

cat_img = io.imread(root + 'cat.jpg')
cat_string = cat_img.tostring()

reconstruct_cat_1d = np.fromstring(cat_string, dtype=np.uint8)
reconstruct_cat_img = reconstruct_cat_1d.reshape(cat_img.shape)
print(np.allclose(cat_img, reconstruct_cat_img))
'''


# metoder
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

root = '/home/jason/test/'
subfolder = 'img'
tfrecords_filename = 'segmentation.tfrecords'

filename_pairs = []
original_images = []

for i in os.listdir(root+subfolder):
    filename_pairs.append(os.path.join(root+subfolder, i))
writer = tf.python_io.TFRecordWriter(root+tfrecords_filename)
print(filename_pairs)

for img_path in filename_pairs:
    img = np.array(Image.open(img_path))
    height = img.shape[0]
    width = img.shape[1]
    original_images.append(img)
    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature=
                                                     {
                                                              'height': _int64_feature(height),
                                                              'width': _int64_feature(width),
                                                              'label': _int64_feature(0),
                                                              'image_raw': _bytes_feature(img_raw),
                                                              'name': _bytes_feature(tf.compat.as_bytes('sopp_1'))

                                                          }))
    writer.write(example.SerializeToString())

writer.close()

reconstructed_images = []
record_iterator = tf.python_io.tf_record_iterator(path='/home/jason/test/'+ tfrecords_filename)

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height']
                 .int64_list
                 .value[0])
    width = int(example.features.feature['width']
                .int64_list
                .value[0])

    img_string = (example.features.feature['image_raw']
                  .bytes_list
                  .value[0])

    img_id = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_id.reshape((height, width, -1))
    reconstructed_images.append(reconstructed_img)

