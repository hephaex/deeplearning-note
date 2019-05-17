import tensorflow as tf
 
# Helper libraries
import numpy as np
import os
 
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]
 
# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(train_images)
 
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
 
EPOCHS = 10
train_steps_per_epoch = len(train_images) // BATCH_SIZE
test_steps_per_epoch = len(test_images) // BATCH_SIZE

with strategy.scope():
  train_iterator = strategy.experimental_make_numpy_iterator(
      (train_images, train_labels), BATCH_SIZE, shuffle=BUFFER_SIZE)
 
  test_iterator = strategy.experimental_make_numpy_iterator(
      (test_images, test_labels), BATCH_SIZE, shuffle=None)
      
def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
 
  return model
  
# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

with strategy.scope():
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
 
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
  model = create_model()
 
  optimizer = tf.keras.optimizers.Adam()
   
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  
  
with strategy.scope():
  # Train step
  def train_step(inputs):
    images, labels = inputs
 
    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = loss_object(labels, predictions)
 
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
 
    train_loss(loss)
    train_accuracy(labels, predictions)
 
  # Test step
  def test_step(inputs):
    images, labels = inputs
 
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
 
    test_loss(t_loss)
    test_accuracy(labels, predictions)
 
 with strategy.scope():
  # `experimental_run` replicates the provided computation and runs it
  # with the distributed input.
   
  @tf.function
  def distributed_train():
    return strategy.experimental_run(train_step, train_iterator)
   
  @tf.function
  def distributed_test():
    return strategy.experimental_run(test_step, test_iterator)
     
  for epoch in range(EPOCHS):
    # Note: This code is expected to change in the near future.
     
    # TRAIN LOOP
    # Initialize the iterator
    train_iterator.initialize()
    for _ in range(train_steps_per_epoch):
      distributed_train()
 
    # TEST LOOP
    test_iterator.initialize()
    for _ in range(test_steps_per_epoch):
      distributed_test()
     
    if epoch % 2 == 0:
      checkpoint.save(checkpoint_prefix)
 
    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                "Test Accuracy: {}")
    print (template.format(epoch+1, train_loss.result(),
                           train_accuracy.result()*100, test_loss.result(),
                           test_accuracy.result()*100))
     
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='eval_accuracy')
 
new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()
 
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

@tf.function
def eval_step(images, labels):
  predictions = new_model(images, training=False)
  eval_accuracy(labels, predictions)

checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
 
for images, labels in test_dataset:
  eval_step(images, labels)
 
print ('Accuracy after restoring the saved model without strategy: {}'.format(
    eval_accuracy.result()*100))

