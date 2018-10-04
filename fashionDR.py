'''
FSU Machine learning 

Eric Dolores

We used Fashion-Mnist to familiarize students with Keras.
'''

from keras.models import Model
from keras import layers
from keras import Input
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping
import numpy as np
from keras.datasets import fashion_mnist
import gc

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0],  28, 28, 1)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

callbacks_list = [
EarlyStopping(monitor = 'acc', patience =1, )
                 ]


image_input = Input(shape = (28,28,1))

block = layers.Conv2D(16, 7, activation = 'relu', padding = 'same')(image_input)
block = layers.Conv2D(16, 7, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(16, 7, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(1, 3, activation = 'relu', padding = 'same', dilation_rate = (1,1))(block)
block = layers.add([block, image_input])

block = layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(1, 3, activation = 'relu', padding = 'same', dilation_rate = (2,2))(block)
block = layers.add([block, image_input])

block = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(1, 3, activation = 'relu', padding = 'same', dilation_rate = (4,4))(block)
block = layers.add([block, image_input])

block = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(1, 3, activation = 'relu', padding = 'same', dilation_rate = (8,8))(block)
block = layers.add([block, image_input])

block = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(block)
block = layers.Conv2D(1, 3, activation = 'relu', padding = 'same', dilation_rate = (16,16))(block)
block = layers.add([block, image_input])


output = layers.Flatten()(block)
output = layers.Dropout(.2)(output)
predictions = layers.BatchNormalization(axis = -1)(output)
predictions = layers.Dense(10,activation = 'softmax' )(predictions)
model = Model(image_input, predictions)
model.summary()

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
model.fit(X_train, y_train, batch_size =300, epochs = 40, callbacks = callbacks_list)
loss, acc = model.evaluate (X_test, y_test)
print('final accuracy = ', acc)
model.save('fashionDR.h5')

print('model saved')
from keras.utils import plot_model
plot_model(model, to_file='fashionDR.png')
gc.collect()
