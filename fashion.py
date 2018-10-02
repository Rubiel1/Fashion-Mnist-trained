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
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0],  28, 28, 1)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

callbacks_list = [
EarlyStopping(monitor = 'acc', patience =2, )
                 ]


image_input = Input(shape = (28,28,1))

branch_a = layers.Conv2D(64, 3, activation='relu')(image_input)
branch_a = layers.Conv2D(128,4, strides = 2, activation='relu' )(branch_a)
branch_a  = layers.BatchNormalization(axis = -1)(branch_a)
branch_b = layers.Conv2D(64, 2, strides = 2, activation='relu')(image_input)
branch_b = layers.Conv2D(128,3, activation = 'relu')(branch_b)
branch_b = layers.BatchNormalization(axis = -1)(branch_b)

output1 = layers.concatenate([branch_a, branch_b], axis = -1)

branch_ab = layers.Conv2D(128, 5, padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l1(.001) )(output1)
branch_c = layers.Conv2D(64,7, padding = 'same', activation='relu')(image_input)
branch_c = layers.Conv2D(128,5,strides = 2 ,  activation = 'relu')(branch_c)
branch_c = layers.BatchNormalization(axis = -1)(branch_c)

output2 = layers.concatenate([branch_c, branch_ab], axis = -1)
output = layers.Conv2D(256,7, padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l1(.001))(output2)
output = layers.Flatten()(output)
output = layers.Dropout(.2)(output)

predictions = layers.Dense(40, activation = 'relu', kernel_regularizer = regularizers.l1(.001))(output)
'''predictions = layers.BatchNormalization(axis = -1)(predictions)'''
predictions = layers.Dropout(.2)(predictions)
predictions = layers.Dense(10,activation = 'softmax' )(predictions)
model = Model(image_input, predictions)
model.summary()

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
model.fit(X_train, y_train, batch_size =300, epochs = 50, callbacks = callbacks_list)
loss, acc = model.evaluate (X_test, y_test)
print('final accuracy = ', acc)
model.save('fashion.h5')
print('model saved')
