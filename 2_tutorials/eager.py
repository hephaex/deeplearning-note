import tensorflow as tf
writer = tf.summary.create_file_writer(logdir="log")
writer.set_as_default()

x = tf.Variable(1, name="x")

for i in range(100):
    x.assign_add(1)
    tf.summary.scalar("x", x, step=i, description="first variable")
