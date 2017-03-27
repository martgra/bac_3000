import tensorflow as tf


def loss(logits, labels):
labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(
        cross_entropy,
        name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')