import os
# CIFAR10_TRAIN.py
MAX_STEPS = 100000
TRAIN_DIR = '/home/jason/tf_train/train_output/'
LOG_DEVICE_PLACEMENT = False
# BATCH_SIZE, found under CIFAR10.py

# CIFAR10_EVAL
EVAL_DIR = '/home/jason/tf_train/train_output/eval'
EVAL_DATA = '/home/jason/tf_train/tfrecords/tfrecords_64/eval'
CHECKPOINT_DIR = '/home/jason/tf_train/train_output'
EVAL_INTERVAL_SECS = 60 * 5
NUM_EXAMPLES = 10000
RUN_ONCE = False
# BATCH_SIZE, found under CIFAR10.py

# CIFAR10_INPUT
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 18000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1600
IMAGE_RAW_SIZE = 2999
IMAGE_DEPTH = 3
# IMAGE_SIZE, found under CIFAR10.py

# CIFAR10
BATCH_SIZE = 32
DATA_DIR = '/home/jason/tf_train/tfrecords/tfrecords_64/train'
USE_FP16 = False
TRAIN_MODE = True

IMAGE_SIZE = 32
 #NUM_CLASSES, found under CIFAR10_INPUT.py
# NUM_EPOCHS_PER_DECAY, found under CIFAR10_INPUT.py
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, found under CIFAR10_INPUT.py
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, found under CIFAR10_INPUT.py

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 25.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.
TOWER_NAME = 'tower'


def hyperwriter():
    with open('/home/jason/tf_train/README.txt', 'w+') as f:
        f.write('# CIFAR10_TRAIN.py \n')
        f.write('MAX_STEPS %d \n' % MAX_STEPS)
        f.write('TRAIN_DIR %s \n' % TRAIN_DIR)
        f.write('LOG_DEVICE_PLACEMENT %r \n' % LOG_DEVICE_PLACEMENT)
        f.write('MAX_STEPS %d \n' % MAX_STEPS)
        f.write('    # BATCH_SIZE, found under CIFAR10.py \n\n')

        f.write('# CIFAR10_EVAL \n')
        f.write('EVAL_DIR %s \n' % EVAL_DIR)
        f.write('EVAL_DATA %s \n' % EVAL_DATA)
        f.write('CHECKPOINT_DIR %s \n' % CHECKPOINT_DIR)
        f.write('EVAL_INTERVAL_SECS %d \n' % EVAL_INTERVAL_SECS)
        f.write('NUM_EXAMPLES %d \n' % NUM_EXAMPLES)
        f.write('NUM_EXAMPLES %r \n' % RUN_ONCE)
        f.write('   # BATCH_SIZE, found under CIFAR10.py \n\n')

        f.write('# CIFAR10_INPUT \n')
        f.write('NUM_CLASSES %d \n' % NUM_CLASSES)
        f.write('NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN %d \n' % NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        f.write('NUM_EXAMPLES_PER_EPOCH_FOR_EVAL %d \n' % NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
        f.write('IMAGE_RAW_SIZE %d \n' % IMAGE_RAW_SIZE)
        f.write('IMAGE_DEPTH %d \n' % IMAGE_DEPTH)
        f.write('    # IMAGE_SIZE, found under CIFAR10.py \n\n')

        f.write('# CIFAR10\n')
        f.write('BATCH_SIZE %d \n' % BATCH_SIZE)
        f.write('DATA_DIR %s \n' % DATA_DIR)
        f.write('USE_FP16 %r \n' % USE_FP16)
        f.write('TRAIN_MODE %r \n' % TRAIN_MODE)
        f.write('IMAGE_SIZE %d \n' % IMAGE_SIZE)
        f.write('   # NUM_CLASSES, found under CIFAR10_INPUT.py \n')
        f.write('   # NUM_EPOCHS_PER_DECAY, found under CIFAR10_INPUT.py \n')
        f.write('   # NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, found under CIFAR10_INPUT.py \n')
        f.write('   # NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, found under CIFAR10_INPUT.py \n')
        f.write('MOVING_AVERAGE_DECAY %d \n' % MOVING_AVERAGE_DECAY)
        f.write('NUM_EPOCHS_PER_DECAY %d \n' % NUM_EPOCHS_PER_DECAY)
        f.write('LEARNING_RATE_DECAY_FACTOR %d \n' % LEARNING_RATE_DECAY_FACTOR)
        f.write('INITIAL_LEARNING_RATE %d \n' % INITIAL_LEARNING_RATE)
        f.write('MOVING_AVERAGE_DECAY %d \n' % MOVING_AVERAGE_DECAY)
        f.write('TOWER_NAME %s \n' % TOWER_NAME)
        f.close()