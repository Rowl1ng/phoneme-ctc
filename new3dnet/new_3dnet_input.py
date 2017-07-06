import tensorflow as tf
import gzip
import cPickle as pickle
import glob
import numpy as np
INPUT_SIZE = 64
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = INPUT_SIZE
  result.width = INPUT_SIZE
  result.depth = INPUT_SIZE
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)
  record_bytes = tf.decode_raw(value, tf.uint8)
  features = tf.parse_single_example(value,
                                     features={
                                         'label': tf.FixedLenFeature([], tf.int64),
                                         'img_raw': tf.FixedLenFeature([], tf.string),
                                     })

  # Convert from a string to a vector of uint8 that is record_bytes long.
  # record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.decode_raw(features['img_raw'], tf.uint8),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result
def load_image(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = INPUT_SIZE
  result.width = INPUT_SIZE
  result.depth = INPUT_SIZE
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.TFRecordReader()
  result.key, value = reader.read(filename_queue)

  features = tf.parse_single_example(value,
                                     features={
                                         'label': tf.FixedLenFeature([], tf.int64),
                                         'img_raw': tf.FixedLenFeature([INPUT_SIZE ** 3], tf.float32),
                                     })

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = features['label']

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  result.uint8image = tf.reshape(
      features['img_raw'],
      [result.depth, result.height, result.width])
  result.uint8image = tf.image.per_image_standardization(result.uint8image)

  # Convert from [depth, height, width] to [height, width, depth].
  # result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, INPUT_SIZE , INPUT_SIZE , INPUT_SIZE] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not tf.gfile.Exists(data_dir):
    print data_dir
    raise ValueError('Failed to find file: ' + data_dir)
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer([data_dir])

  # Read examples from files in the filename queue.
  read_input  = load_image(filename_queue)
  # read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)


  # Subtract off the mean and divide by the variance of the pixels.
  # float_image = tf.image.per_image_standardization(reshaped_image)
  # Set the shapes of tensors.
  reshaped_image.set_shape([INPUT_SIZE , INPUT_SIZE , INPUT_SIZE])
  # read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not tf.gfile.Exists(data_dir):
    print data_dir
    raise ValueError('Failed to find file: ' + data_dir)
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer([data_dir])

  # Read examples from files in the filename queue.
  read_input  = load_image(filename_queue)
  # read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)


  # Subtract off the mean and divide by the variance of the pixels.
  # float_image = tf.image.per_image_standardization(reshaped_image)
  # Set the shapes of tensors.
  reshaped_image.set_shape([INPUT_SIZE , INPUT_SIZE , INPUT_SIZE])
  # read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

# distorted_inputs("",10)
# data_dir = '/home/luoling/phoneme/new3dnet/train.tfrecords'
# if not tf.gfile.Exists(data_dir):
#   print data_dir
#   raise ValueError('Failed to find file: ' + data_dir)
# else :
#   print 'OK'
