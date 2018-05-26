
# net inport
import numpy as np
import keras.backend
from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import mnist
np.random.seed(123)

# generator -> (X_text, Y_test)

# запилим модель с блекджеком и ...
# когда буду накидывать рекурентные последовательности должны быть stateful
# reset recurent будет звучать как-то как model.reset_states
import PythonClient.airsimWithNet as airsimdata
batch_size = 1
temp_data_x, temp_data_y = airsimdata.processDataForSavingAndForNet()
shape_temp_x = temp_data_x.shape
shape_temp_y = temp_data_y.shape
print(shape_temp_x, shape_temp_y)
model = Sequential()
keras.backend.set_image_data_format("channels_last")
# model.add(InputLayer(temp_data_x.shape[1:]))
model.add(Convolution2D(8, (3,3), activation='relu', input_shape=(480, 640, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(480*640))
model.add(Reshape((480,640)))

model.compile(loss='categorical_crossentropy',
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



while (ep < 10):
    try:
        X_test, Y_test = generator
        # model.fit_generator(generator,epochs=epochs, steps_per_epoch=1, verbose=1, workers=0) #
        model.fit(X_test, Y_test, 1, 1)
    except airsimdata.ExeptInGenData as ex:
        model.reset_states()
    finally: ep += 1
print("<3")
# подкоректируем форму данных
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

# тренировка 1 итерацию?
# model.fit(X_train, Y_train, batch_size=1, nb_epoch=1, verbose=1)


# model.fir_generator(...generator, sdsd)

# end of cicle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# проверка результата на тестовых данных
# score = model.evaluate(X_test, Y_test, verbose=1)
# print(score)