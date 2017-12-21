from astropy.stats.funcs import _check_poisson_conf_inputs
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
#inputs = Input(shape=(784,))
inputs = Input(shape=(4,))


# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#model.fit(data, labels)  # starts training
