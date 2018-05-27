# net import
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
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

model.add(ConvLSTM2D(16, (3, 3), activation='relu', batch_input_shape=(1, 1, ROWS, COLS, 1), return_sequences=True, stateful=True))

model.add(ConvLSTM2D(8, (3, 3), activation='relu', return_sequences=True, stateful=True))

model.add(ConvLSTM2D(8, (3, 3), activation='relu', stateful=True))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(AveragePooling2D(pool_size=(3, 3)))


model.add(Flatten())
model.add(Dense(ROWS * COLS, activation='sigmoid'))
model.add(Reshape((ROWS, COLS)))

opt = keras.optimizers.Adam(lr= 0.0005)
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])
print(model.summary())
# data import

def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    —-------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def generator():
  while True:
    yield airsimdata.processDataForSavingAndForNet()

# cicle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# ((X_train, Y_train), reset) = airsimdata.getData()
# print("Shape of x: ",X_train.shape, ", shape of Y ", Y_train.shape)

# получим данные
# ((X_train, Y_train), reset) = airsimdata.getData()
# немного о данных
# Shape of x:  (480, 640, 4) , shape of Y  (480, 640)
epochs = 1
ep = 0


def testmodel(epoch, logs):
    predx, predy = next(generator())

    predout = model.predict(
        predx,
        batch_size=1
    )
    print(predx)
    print(predy)
    print(predout)
    #plt.imshow(predx)
    #plt.show()
    #plt.imshow(predy)
    #plt.show()
    #plt.imshow(predout)
    #plt.show()
    #show_images([predx, predy, predout], 1, ["get", "want", "predict"])

MyTensorBoardDir = "L:\\Documents\\PyCharmProjects\\HelloDrone\\logs"
testmodel_cb = keras.callbacks.LambdaCallback(on_epoch_end=testmodel)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=MyTensorBoardDir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

while ep < 30:
  try:
    model.fit_generator(generator(), epochs=epochs, steps_per_epoch=1000, verbose=1, workers=1)
    x_data, y_data = next(generator())
    res = model.predict(x_data)
    show_images([np.reshape(x_data, (ROWS, COLS)), np.reshape(y_data, (ROWS, COLS)), np.reshape(res,(ROWS, COLS)),
                 np.reshape(y_data, (ROWS, COLS)) -  np.reshape(res,(ROWS, COLS))], 1, ["from", "want", "predict", "diff"])
    airsimdata.movementConnection()
  except airsimdata.ExeptInGenData as ex:
    model.reset_states()
  finally: ep += 1
model.save('model.h5')

print("<3")