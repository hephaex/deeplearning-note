CNN을 tensorflow로 구현한 것을 정리합니다. 

# CNN
## CNN이란
　CNN은 주로 이미지 인식이나 이미지 분류 등의 작업에 사용되는 신경망을 말합니다. 회귀층과 풀링층, 전결합층이라는 층을 가지고 있는 것이 특징입니다.
　Convolutional Neural Network의 약자로, 우리말로는 컨볼루션 신경망이라고 한다.
 
## CNN 구현
tensorflow에 Conv2D가 있으므로 이를 사용한다.
```py
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

## Alex net
AlexNet은 2012년 인공지능 이미지 인식 대회에서 우승한 모델입니다. 4번째 AI의 봄, 딥러닝의 시작인 모델입니다. 현재 AI 붐/딥러닝 붐의 시발점이 된 모델이다.
[AlexNet-CNN[(https://www.researchgate.net/figure/An-illustration-of-the-architecture-of-AlexNet-CNN-14_fig4_312188377)

```py
def alexet(input_shape=(227, 227, 3), num_classes=1000):
    model = models.Sequential()

    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))

    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))

    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
```
    
