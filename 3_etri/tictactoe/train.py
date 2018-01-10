# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import tensorflow as tf


def inference(x_ph):
    hidden1 = tf.layers.dense(x_ph, 32, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, 32, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, 3)
    return logits


np.random.seed(1)

mat = np.loadtxt("data/tictactoe.csv", skiprows=1, delimiter=",")
ind_train = np.random.choice(5890, 4000, replace=False)
ind_test = np.array([i for i in range(5890) if i not in ind_train])

x_train = mat[ind_train, :-1]
x_test = mat[ind_test, :-1]
y_all = np.zeros([len(mat), 3])
for i, j in enumerate(mat[:, -1]):
    if j == 1:
        # x win
        y_all[i][0] = 1.
    elif j == -1:
        # o win
        y_all[i][1] = 1.
    else:
        # draw
        y_all[i][2] = 1.
y_train = y_all[ind_train]
y_test = y_all[ind_test]

with tf.Graph().as_default() as g:
    tf.set_random_seed(0)
    x_ph = tf.placeholder(tf.float32, [None, 9])
    y_ph = tf.placeholder(tf.float32, [None, 3])
    logits = inference(x_ph)
    y = tf.nn.softmax(logits)
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_ph, logits=logits, label_smoothing=1e-5)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(10000):
            ind = np.random.choice(len(y_train), 1000)
            sess.run(train_op, feed_dict={x_ph: x_train[ind], y_ph: y_train[ind]})
            if i % 100 == 0:
                train_loss = sess.run(cross_entropy, feed_dict={x_ph: x_train, y_ph: y_train})
                train_accuracy, y_pred = sess.run([accuracy, y], feed_dict={x_ph: x_train, y_ph: y_train})
                test_accuracy = sess.run(accuracy, feed_dict={x_ph: x_test, y_ph: y_test})
                print(
                    "Iteration: {0} Loss: {1} Train Accuracy: {2} Test Accuracy{3}".format(
                        i, train_loss, train_accuracy, test_accuracy
                    )
                )
        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")
        saver.save(sess, "checkpoints/tictactoe")
        # Remove old model
        if os.path.exists("model"):
            shutil.rmtree("model")
        # Save model for deployment on ML Engine
        input_key = tf.placeholder(tf.int64, [None, ], name="key")
        output_key = tf.identity(input_key)
        input_signatures = {
            "key": tf.saved_model.utils.build_tensor_info(input_key),
            "x": tf.saved_model.utils.build_tensor_info(x_ph)
        }
        output_signatures = {
            "key": tf.saved_model.utils.build_tensor_info(output_key),
            "y": tf.saved_model.utils.build_tensor_info(y)
        }
        predict_signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            input_signatures,
            output_signatures,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join("model"))
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def
            },
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
        )
        builder.save()
