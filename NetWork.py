
# net inport
import numpy as np
import keras.backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
np.random.seed(123)

# generator -> (X_text, Y_test)

# запилим модель с блекджеком и ...
# когда буду накидывать рекурентные последовательности должны быть stateful
# reset recurent будет звучать как-то как model.reset_states

model = Sequential()
keras.backend.set_image_data_format("channels_first")
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',''])

# data import
import PythonClient.airsimWithNet as airsimdata
batch_size = 30
generator = (airsimdata.getData() for i in range(batch_size))
# cicle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

# получим данные
# ((X_train, Y_train), reset) = airsimdata.getData()
# немного о данных
# X -
epochs = 1
model.fit_generator(generator,batch_size,epochs, verbose=1)

# подкоректируем форму данных
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

# тренировка 1 итерацию?
# model.fit(X_train, Y_train, batch_size=1, nb_epoch=1, verbose=1)


# model.fir_generator(...generator, sdsd)

# end of cicle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# проверка результата на тестовых данных
# score = model.evaluate(X_test, Y_test, verbose=1)
# print(score)