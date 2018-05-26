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

plt.imshow(np.reshape(temp_data_x, (128, 128)), cmap="gray")
plt.show()
plt.imshow(np.reshape(temp_data_y, (128, 128)), cmap="gray")
plt.show()

model = Sequential()
K.set_image_data_format("channels_last")

model.add(ConvLSTM2D(16, (3, 3), activation='relu', batch_input_shape=(1, 1, 128, 128, 1), return_sequences=True, stateful=True))

model.add(ConvLSTM2D(16, (3, 3), activation='relu', return_sequences=True, stateful=True))

model.add(ConvLSTM2D(16, (3, 3), activation='relu', stateful=True))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(AveragePooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(128 * 128, activation='sigmoid'))
model.add(Reshape((128, 128)))

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
# подкоректируем форму данных
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

# тренировка 1 итерацию?
# model.fit(X_train, Y_train, batch_size=1, nb_epoch=1, verbose=1)


# model.fir_generator(...generator, sdsd)

# end of cicle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# проверка результата на тестовых данных
# score = model.evaluate(X_test, Y_test, verbose=1)
# print(score)
model.save('model.h5')