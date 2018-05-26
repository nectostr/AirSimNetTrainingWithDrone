# net import
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
np.random.seed(123)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
K.set_session(sess)
ROWS = 150
COLS = 200
# generator -> (X_text, Y_test)

# запилим модель с блекджеком и ...
# когда буду накидывать рекурентные последовательности должны быть stateful
# reset recurrent будет звучать как-то как model.reset_states
import PythonClient.airsimWithNet as airsimdata
batch_size = 1
temp_data_x, temp_data_y = airsimdata.processDataForSavingAndForNet()
shape_temp_x = temp_data_x.shape
shape_temp_y = temp_data_y.shape
print(shape_temp_x, shape_temp_y)

model = Sequential()
K.set_image_data_format("channels_last")

model.add(ConvLSTM2D(8, (3, 3), activation='relu', batch_input_shape=(1, 1, ROWS, COLS, 1), return_sequences=True, stateful=True))

model.add(ConvLSTM2D(8, (3, 3), activation='relu', return_sequences=True, stateful=True))

model.add(ConvLSTM2D(8, (3, 3), activation='relu', stateful=True))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(AveragePooling2D(pool_size=(5, 5)))


model.add(Flatten())
model.add(Dense(ROWS * COLS, activation='sigmoid'))
model.add(Reshape((ROWS, COLS)))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
# data import



def generator():
  while True:
    yield airsimdata.getData()

# cicle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# ((X_train, Y_train), reset) = airsimdata.getData()
# print("Shape of x: ",X_train.shape, ", shape of Y ", Y_train.shape)

# получим данные
# ((X_train, Y_train), reset) = airsimdata.getData()
# немного о данных
# Shape of x:  (480, 640, 4) , shape of Y  (480, 640)
epochs = 1
ep = 0



while ep < 3:
  try:
    model.fit_generator(generator(), epochs=epochs, steps_per_epoch=100, verbose=1, workers=1) #
  except airsimdata.ExeptInGenData as ex:
    model.reset_states()
  finally: ep += 1
model.save('model.h5')

print("<3")