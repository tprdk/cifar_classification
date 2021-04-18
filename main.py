import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

#params
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10

#loading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(f'x_train shape : {x_train.shape} - x_test shape : {x_test.shape}')

#Normalization
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

#Building model
model = keras.Sequential()
model.add(layers.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu',))
model.add(layers.Conv2D(64, kernel_size=3, activation='relu',))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, kernel_size=3, activation='relu',))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(10, activation='softmax'))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
print('Evaluate : ')
model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)