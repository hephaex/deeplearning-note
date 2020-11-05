#!/usr/bin/env python
# coding: utf-8

# # Image data and augmentation using Keras

# ### Importing Libraries

# In[1]:


import os
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

print('Using Tensorflow',tf.__version__)


# #### Instantiating

# In[2]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40)


# In[3]:


image_path = 'S:/Project Final Year/Testing images/Images/4.jpg'
path = 'S:/Project Final Year/Testing images'
plt.imshow(plt.imread(image_path))


# In[4]:


x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Rotation

# In[5]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40
)


# In[6]:


x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Width and Height Shifts

# In[7]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=[-100,-50,0,50,100],
    height_shift_range = [-50,0,50]
)


# In[8]:


x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Brightness

# In[9]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    brightness_range=(0.5,2)
)
x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Shear transformation

# In[10]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    shear_range=40
)
x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Zoom

# In[11]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.6
)
x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Channel Shift

# In[12]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    channel_shift_range=100
)
x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# In[13]:


x.mean()


# In[14]:


np.array(Image.open(image_path)).mean()


# ## Flips

# In[15]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)
x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Multiple augmentations

# In[16]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=50
)
x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# In[ ]:





# In[17]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=50
)
x,y = next(generator.flow_from_directory('images'))
plt.imshow(x[0].astype('uint8'))


# ## Normalization

# In[ ]:


x_train, y_train, x_test, y_test  = tf.keras.datasets.cifar10.load_data()
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    fearurewise_std_normalization = True
)

generator.fit(x_train)


# In[ ]:


x,y = next(generator.flow(x_train, y_train))
print(x.mean(), x.std(), y)
print(x_train.mean())


# ## Rescaling and Normalization

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1.,
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
    horizontal_flip=True, rotation_range=20
)


# ## Using in Model Training

# In[87]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1.,
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
    horizontal_flip=True, rotation_range=20
)


# In[94]:


model = tf.keras.models.Sequential([
    tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False, input_shape=(96,96,3),
        pooling='avg'
    ),
    tf.keras.layers.Dense(10,activation='softmax')
])     

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrices=['accuracy']
)


# In[ ]:


model = model.fit(
    generator.flow(x_train,y_train,epochs=1, steps_per_epoch=10)
)


# In[ ]:




