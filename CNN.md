
# Alex net
'''python
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
    '''
    
