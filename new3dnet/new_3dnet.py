import numpy as np
import tensorflow as tf
import re
import new_3dnet_input

FC_SIZE = 420
DTYPE = tf.float32
TOWER_NAME = 'tower'
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 12,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/large_train.bin',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
# IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5000.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.3       # Initial learning rate.

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = new_3dnet_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                    batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = new_3dnet_input.inputs(eval_data=eval_data,
                                          data_dir=eval_data,
                                          batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels
def inference(boxes, training = False):
    """Build the 3Dnet model.
    Args:
      boxes: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    prev_layer = boxes

    in_filters = 1 #dataconfig.num_props
    with tf.variable_scope('conv1') as scope:
        out_filters = 64
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 5, in_filters, out_filters],
                                             stddev=1e-6,
                                             wd=0.0)
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_filters],  tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        # norm4 = tf.contrib.layers.batch_norm(bias, center=True, is_training=training, scale=True)
        conv1 = tf.nn.relu(bias, name=scope.name)
        conv1 = tf.layers.dropout(inputs= conv1, rate=0.2, training=training)
        _activation_summary(conv1)
        prev_layer = conv1
        in_filters = out_filters

    pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    # norm1 = tf.contrib.layers.batch_norm(pool1, data_format='NDHWC', center=True,is_training = True, scale =True)

    prev_layer = pool1

    with tf.variable_scope('conv2') as scope:
        out_filters = 64
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 5, in_filters, out_filters],
                                             stddev=1e-6,
                                             wd=0.0)
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_filters],  tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        conv2 = tf.layers.dropout(inputs= conv2, rate=0.2, training=training)
        _activation_summary(conv2)
        prev_layer = conv2
        in_filters = out_filters

    # normalize prev_layer here
    # prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    # with tf.variable_scope('conv3') as scope:
    #     out_filters = 96
    #     kernel = _variable_with_weight_decay('weights',
    #                                          shape=[5, 5, 5, in_filters, out_filters],
    #                                          stddev=5e-2,
    #                                          wd=0.0)
    #     conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
    #     biases = _variable_on_cpu('biases', [out_filters], tf.constant_initializer(0.1))
    #     bias = tf.nn.bias_add(conv, biases)
    #     norm3 = tf.contrib.layers.batch_norm(bias, center=True, is_training=training, scale=True)
    #     conv3 = tf.maximum(0.01 * norm3, norm3)
    #     # conv3 = tf.nn.relu(norm3, name=scope.name)
    #     _activation_summary(conv3)
    #     prev_layer = conv3

    with tf.variable_scope('conv3_1') as scope:
        out_filters = 64
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 5, in_filters, out_filters],
                                             stddev=1e-6,
                                             wd=0.0)
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_filters],  tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        prev_layer = tf.layers.dropout(inputs= prev_layer, rate=0.2, training=training)
        _activation_summary(prev_layer)
        in_filters = out_filters

    with tf.variable_scope('conv3_2') as scope:
        out_filters = 64
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 5, in_filters, out_filters],
                                             stddev=1e-6,
                                             wd=0.0)
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_filters],  tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        prev_layer = tf.layers.dropout(inputs= prev_layer, rate=0.2, training=training)
        _activation_summary(prev_layer)
        in_filters = out_filters

    with tf.variable_scope('conv3_3') as scope:
        out_filters = 32
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 5, in_filters, out_filters],
                                             stddev=1e-6,
                                             wd=0.0)
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_filters],  tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        prev_layer = tf.layers.dropout(inputs= prev_layer, rate=0.2, training=training)

        _activation_summary(prev_layer)
        in_filters = out_filters

    pool2 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    prev_layer = pool2
    with tf.variable_scope('local4') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, FC_SIZE],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [FC_SIZE], tf.constant_initializer(0.1))
        bias = tf.matmul(prev_layer_flat, weights) + biases
        bias = tf.contrib.layers.batch_norm(bias, center=True, is_training=training, scale=True)

        # local4 = tf.maximum(0.01 * bias, bias)

        local4 = tf.nn.relu(bias, name=scope.name)
        local4 = tf.layers.dropout(inputs= local4, rate=0.2, training=training)
        _activation_summary(local4)

    prev_layer = local4

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        weights = _variable_with_weight_decay('weights', [dim, NUM_CLASSES], stddev=1/192.0, wd=0.0)#dataconfig.num_classes])
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))#dataconfig.num_classes])
        softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')

    # return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op