import tensorflow as tf

input = [
  [1., 0., 0.],
  [0., 1., 0.],
  [0., 0., 1.]
]

winning_hands = [
  [0., 1., 0.],
  [0., 0., 1.],
  [1., 0., 0.]
]

x = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.zeros( [3, 3]))
b = tf.Variable(tf.zeros( [3]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 3])
cross_entropy = -tf.reduce_sum( y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init =tf.initialize_all_variables() 
feed_dict = {x: input, y_: winning_hands}

with tf.Session() as sess:
  sess.run(init)

  for step in range(1000):
    sess.run(train_step, feed_dict = feed_dict)
    if step % 100 == 0:
      print sess.run(cross_entropy, feed_dict=feed_dict)
