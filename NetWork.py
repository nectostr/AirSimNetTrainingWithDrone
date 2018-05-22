
# net inport
import numpy as np
import keras.backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPool3D, Reshape
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
model.add(InputLayer(temp_data_x.shape))
"""model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))"""
model.add(Flatten())
model.add(Dense(temp_data_x.shape[0]*temp_data_x.shape[1]*temp_data_x.shape[2], activation="sigmoid"))
model.add(Dense(temp_data_x.shape[0]*temp_data_x.shape[1]))
model.add(Reshape(shape_temp_y))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# data import



generator = (airsimdata.getData() for i in range(10))
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
        model.fit_generator(generator,epochs=epochs, verbose=1, workers=0) #
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