from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime

import tensorflow as tf

import calculate_loss
import read_and_batch
import train


def train():
    tfrecords_filename = '/home/jason/test/segmentation.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, labels = read_and_batch.read_and_decode(filename_queue)
        logtis = train.martin_net(images)
        loss = calculate_loss.loss(logtis, images)
        train_op = train.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self.step % 10 == 0:
                    num_examples_per_step = 32
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        with tf.train.MonitoredSession(
            checkpoint_dir ='/home/jason/',
            hooks=[tf.train.StopAtStepHook(last_step=10000),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


train()
tf.app.run()






'''
tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
    tf.app.run()


def main(unused_argv):
    tfrecords_filename = '/home/jason/test/segmentation.tfrecords'
    queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=10)
    train_data, train_labels = read_and_batch(queue)
    eval_data, eval_labels = read_and_batch(queue)

    MartinNet = learn.Estimator(
        model_fn=model.MartinNet,
        model_dir="/tmp/MartinNet")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50)

    MartinNet.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        monitors=[logging_hook])

    metrics = {
        "accuracy":
            learn.metric_spec.MetricSpec(
                metric_fn=tf.metrics.accuracy, preciction_key="classes"),
    }

    eval_results = mnist_classifier.evaluate(
        x=eval_data,
        y=eval_labels,
        metrics=metrics)
    print(eval_results)
'''