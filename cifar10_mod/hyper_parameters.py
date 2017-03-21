# CIFAR10_TRAIN.py
MAX_STEPS = 1000000
TRAIN_DIR = '/tmp/cifar10_train'
LOG_DEVICE_PLACEMENT = False
# BATCH_SIZE, found under CIFAR10.py

# CIFAR10_EVAL
EVAL_DIR = '/tmp/cifar10_eval'
EVAL_DATA = 'test'
CHECKPOINT_DIR = '/tmp/cifar10_train'
EVAL_INTERVAL_SECS = 60 * 5
NUM_EXAMPLES = 10000
RUN_ONCE = False
# BATCH_SIZE, found under CIFAR10.py

# CIFAR10_INPUT
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
TFRECORDS_DIR = '/home/jason/tf_train/tfrecords2'
IMAGE_RAW_SIZE = 299
IMAGE_DEPTH = 3
# IMAGE_SIZE, found under CIFAR10.py

# CIFAR10
BATCH_SIZE = 128
DATA_DIR = '/home/jason/tf_train/tfrecords2'
USE_FP16 = False

IMAGE_SIZE = 24
# NUM_CLASSES, found under CIFAR10_INPUT.py
# NUM_EPOCHS_PER_DECAY, found under CIFAR10_INPUT.py
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, found under CIFAR10_INPUT.py
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, found under CIFAR10_INPUT.py

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
TOWER_NAME = 'tower'