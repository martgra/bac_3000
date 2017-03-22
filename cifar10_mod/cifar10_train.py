# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10
from hyper_parameters import *

#MAX_STEPS = hyper_parameters.MAX_STEPS
#TRAIN_DIR = hyper_parameters.TRAIN_DIR
#LOG_DEVICE_PLACEMENT = hyper_parameters.LOG_DEVICE_PLACEMENT
#BATCH_SIZE = hyper_parameters.BATCH_SIZE

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.

        logits = cifar10.inference(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)\n')
                    print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
                    with open('/home/jason/tf_train/train_output/test.txt', 'a+') as f:
                        f.write(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(checkpoint_dir=TRAIN_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS), tf.train.NanTensorHook(loss), _LoggerHook()],
                config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    if tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.DeleteRecursively(TRAIN_DIR)
    tf.gfile.MakeDirs(TRAIN_DIR)
    train()


if __name__ == '__main__':
    hyperwriter()
    tf.app.run()
