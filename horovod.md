## install horovod

```
HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
```

```
import os
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import time
from tensorflow import keras

layers = tf.layers
tf.logging.set_verbosity(tf.logging.INFO)

def conv_model(feature, target, mode):

    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    with tf.variable_scope('conv_layer1'):
        h_conv1 = layers.conv2d(feature, 32, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv_layer2'):
        h_conv2 = layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    h_fc1 = layers.dropout(
        layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu),
        rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = layers.dense(h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    tf.summary.scalar('loss', loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1),
                                  tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    return tf.argmax(logits, 1), loss, accuracy

def train_input_generator(x_train, y_train, batch_size=64):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size

def main(_):
    start_time = time.time()

    hvd.init()

    (x_train, y_train), (x_test, y_test) = \
        keras.datasets.mnist.load_data('MNIST-data-%d' % hvd.rank())


    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    x_test = np.reshape(x_test, (-1, 784)) / 255.0

    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32, [None, 784], name='image')
        label = tf.placeholder(tf.float32, [None], name='label')
    predict, loss, accuracy = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)

    opt = tf.train.AdamOptimizer(0.001 * hvd.size())

    opt = hvd.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.StopAtStepHook(last_step=20000 // hvd.size()),
        tf.train.LoggingTensorHook(tensors={'step': global_step,
                       'loss': loss,
                       'accuracy':accuracy},
                                   every_n_iter=10),
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    checkpoint_dir = './new/checkpoints' if hvd.rank() == 0 else None
    training_batch_generator = train_input_generator(x_train,
                                                     y_train, batch_size=100)

    merged = tf.summary.merge_all()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
         writer = tf.summary.FileWriter("./tflog/%d" % hvd.rank(), mon_sess.graph)
         while not mon_sess.should_stop():
             image_, label_ = next(training_batch_generator)
             _, result, step = mon_sess.run([train_op,merged,global_step], feed_dict={image: image_, label: label_})
             if step % 100 == 0:
                 writer.add_summary(result,step)
    duration = time.time()-start_time
    print("/device:GPU:" + str(hvd.local_rank()) + "runtime is --- %s seconds ---" % duration)

if __name__ == "__main__":
    tf.app.run()
```
