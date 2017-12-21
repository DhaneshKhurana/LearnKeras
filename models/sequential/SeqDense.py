from keras.models import Sequential
from keras.models import Layer
from keras.layers import Dense
from keras import losses as l
import keras as k
import numpy as np


model = Sequential()
model.add(Dense(units=64, activation="relu", input_dim=100))
model.add(Dense(units=100, activation="softmax"))

#model.compile(optimizer=k.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), loss=l.categorical_crossentropy, metrics="[accuracy]")

#model.fit(x, y, epochs=5, batch_size=32)

data = np.random.random((10, 5))


print(data)

x_train = np.empty((1, 3, 4, 2), dtype='uint8')

print(x_train)