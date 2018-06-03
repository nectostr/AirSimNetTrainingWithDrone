# net import
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
from keras.regularizers import *
from keras.constraints import *
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
ROWS = 64
COLS = 64


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
    i = 0
    while True:
        x = np.loadtxt("L:\\Documents\\PyCharmProjects\\HelloDrone\\data\\pic_from" + str(i) + ".txt")
        y = np.loadtxt("L:\\Documents\\PyCharmProjects\\HelloDrone\\data\\pic_to" + str(i) + ".txt")
        x = np.expand_dims(np.expand_dims(x, 0),-1)
        y = np.expand_dims(np.expand_dims(y, 0), -1)
        if i == 2005: i = -1
        i += 1
        yield x,y


model = keras.models.load_model('model2.h5')

epochs = 1
ep = 0
a = generator()
while ep < 3:

    try:
        print(ep)
        history = model.fit_generator(a, epochs=epochs, steps_per_epoch=300, verbose=1, workers=1)
        x_data, y_data = next(a)
        res = model.predict(x_data)
        show_images([np.reshape(x_data, (ROWS, COLS)), np.reshape(y_data, (ROWS, COLS)), np.reshape(res,(ROWS, COLS)),
                    ], 1, ["from", "want", "predict"])
        #airsimdata.resetImageConn()
        model.reset_states()
        model.save('model2' + ep +'.h5')
    except airsimdata.ExeptInGenData as ex:
        model.reset_states()
    finally:
        ep += 1
print(history.history['loss'])
for i in range(10):
    for j in range(10):
        x_data, y_data = next(generator())
        res = model.predict(x_data)
    show_images([np.reshape(x_data, (ROWS, COLS)), np.reshape(y_data, (ROWS, COLS)), np.reshape(res,(ROWS, COLS)),
                ], 1, ["from", "want", "predict"])
model.save('model.h5')

print("<3")
