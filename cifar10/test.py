#-*- coding:utf-8 -*
import tensorflow as tf
import gzip
import cPickle as pickle
import glob
import numpy as np
import cifar10
from new3dnet import new_3dnet_input
INPUT_SIZE = 64
data_dir = './candidates_v2_0.75mm_64_3d/train2_new_tianci_subset[0-7]/*/*.pkl.gz'
def normalize(image):
    image = (image - (-1000.0)) / (400.0 - (-1000.0))
    image[image>1] = 1.
    image[image<0] = 0.
    return image
def create_record():
    '''
    此处我加载的数据目录如下：
    0 -- img1.jpg
         img2.jpg
         img3.jpg
         ...
    1 -- img1.jpg
         img2.jpg
         ...
    2 -- ...
    ...
    '''

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    label_bytes = 1  # 2 for CIFAR-100
    result.height = 64
    result.width = 64
    result.depth = 64
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    filenames = glob.glob(data_dir)
    writer = tf.python_io.TFRecordWriter("/tmp/train.bin")
    for image_path in filenames:
        with gzip.open(image_path) as file:
            x = pickle.load(file)
            x = normalize(x).flatten()
            image = np.asarray(x, np.uint8)
            label = 1 if "True" in image_path else 0
            img_raw = image.tobytes() #将图片转化为原生bytes
            print len(img_raw)
            features = np.zeros(64**3)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=x.astype("float")))
            }))
            # out = np.array(list(label) + list(image.flatten()), np.uint8)
            writer.write(example.SerializeToString())
    writer.close()

def read_and_decode_single(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([64**3], tf.float32),
                                       })
    image = features['img_raw']
    label = features['label']
    image = tf.reshape(image, [64, 64, 64])
    return image, label

def read_and_decode(filename):
    print filename
    filename_queue = tf.train.string_input_producer([filename])

    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # features = tf.parse_single_example(serialized_example,
    #                                    features={
    #                                        'label': tf.FixedLenFeature([], tf.int64),
    #                                        'img_raw' : tf.FixedLenFeature([], tf.string),
    #                                    })
    #
    # img = tf.decode_raw(features['img_raw'], tf.uint8)
    # img = tf.reshape(img, [64, 64, 64])
    # # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # label = tf.cast(features['label'], tf.int32)
    result = read_cifar10(filename_queue)
    image = tf.cast(result.uint8image, tf.float32)
    image.set_shape([32, 32, 3])
    result.label.set_shape([1])

    return image, result.label
    # return img, label
def read_and_decode_tianchi(filename):
    print filename
    filename_queue = tf.train.string_input_producer([filename])

    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # features = tf.parse_single_example(serialized_example,
    #                                    features={
    #                                        'label': tf.FixedLenFeature([], tf.int64),
    #                                        'img_raw' : tf.FixedLenFeature([], tf.string),
    #                                    })
    #
    # img = tf.decode_raw(features['img_raw'], tf.uint8)
    # img = tf.reshape(img, [64, 64, 64])
    # # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # label = tf.cast(features['label'], tf.int32)
    result = read_tianchi(filename_queue)
    image = tf.cast(result.uint8image, tf.float32)
    image.set_shape([64, 64, 64])
    result.label.set_shape([1])

    return image, result.label
def read_and_decode_tianchi_test(filename):
    print filename
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 64])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label
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
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result
def read_tianchi(filename_queue):
  class tianchiRecord(object):
    pass
  result = tianchiRecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 64
  result.width = 64
  result.depth = 64
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result

np.set_printoptions(threshold='nan')
if __name__ == '__main__':
    # create_record()

    # quit()
    # img, label = read_and_decode_tianchi("/tmp/train.bin")
    # img, label = read_and_decode_tianchi_test("/tmp/train.tfrecords")

    img, label = read_and_decode("/tmp/cifar10_data/cifar-10-batches-bin/data_batch_1.bin")
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=3, capacity=3,
                                                    min_after_dequeue=2)
    # img_batch, label_batch = new_3dnet_input.distorted_inputs(data_dir="/tmp/train.bin",
    #                                                   batch_size=3)
    global_step = tf.contrib.framework.get_or_create_global_step()

    y_pred = cifar10.inference(img_batch)
    # label_batch = tf.cast(label_batch, tf.float32)
    # loss = tf.nn.l2_loss(y_pred-label_batch)
    label_batch = tf.cast(label_batch, tf.int64)
    label_batch = tf.reshape(label_batch, [3])
    # label_batch = tf.reduce_sum(label_batch, axis=-1)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=y_pred, labels=label_batch, name='cross_entropy_per_example')
    loss = cifar10.loss(y_pred, label_batch)
    # loss_mean = tf.reduce_mean(loss)

    # train_op = tf.train.AdamOptimizer().minimize(loss)
    train_op = cifar10.train(loss, global_step)
    #初始化所有的op
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        #启动队列
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(0,3):
            # val, l= sess.run([img_batch, label_batch])
            _,y_pred = sess.run([train_op,y_pred])
            #l = to_categorical(l, 12)
            # print(val.shape, l)
            print y_pred
            # print loss_val
            print("Worked!!!")