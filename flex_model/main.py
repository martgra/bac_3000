#from tensorflow.examples.tutorials.mnist import input_data
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flex_input import *
import tensorflow as tf

url = 'http://commondatastorage.googleapis.com/books1000/'
data_root = './notMNIST'
train_filename_tar = 'notMNIST_large.tar.gz'
test_filename_tar = 'notMNIST_small.tar.gz'
bytes_train = 247336696
bytes_test = 8458043
num_classes = 10
image_size = 28
pixel_depth = 255.0
num_labels = 10
num_channels = 1


train_filename = download_notmnist(train_filename_tar, url, data_root,  bytes_train)
test_filename = download_notmnist(test_filename_tar, url, data_root, bytes_test)

train_folders = extract_notmnist(data_root, train_filename, num_classes)
test_folders = extract_notmnist(data_root, test_filename, num_classes)

train_datasets = maybe_pickle(train_folders, 45000, image_size, pixel_depth)
test_datasets = maybe_pickle(test_folders, 1800, image_size, pixel_depth)

train_size = 500000
valid_size = 1000
test_size = 1000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, image_size,
                                                                          valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size, image_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

train_dataset1, train_labels1 = reformat(train_dataset, train_labels, image_size, num_labels)
valid_dataset1, valid_labels1 = reformat(valid_dataset, valid_labels, image_size, num_labels)
test_dataset1, test_labels1 = reformat(test_dataset, test_labels, image_size, num_labels)

print('Training set', train_dataset1.shape, train_labels1.shape)
print('Validation set', valid_dataset1.shape, valid_labels1.shape)
print('Test set', test_dataset1.shape, test_labels1.shape)


train_dataset, train_labels = reformat2(train_dataset, train_labels, image_size, num_labels, num_channels)
valid_dataset, valid_labels = reformat2(valid_dataset, valid_labels, image_size, num_labels, num_channels)
test_dataset, test_labels = reformat2(test_dataset, test_labels, image_size, num_labels, num_channels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)




#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4,).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
batch_size = 50

for step in range(200000):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    if step%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g"% (step, train_accuracy))
        train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.50})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_dataset, y_: test_labels, keep_prob: 1.0}))