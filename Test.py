# net import
import numpy as np
import tensorflow as tf
from keras.layers import *
from matplotlib import pyplot as plt
np.random.seed(123)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
K.set_session(sess)
ROWS = 256
COLS = 256


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
        x = np.load("C:\\data12\\pic_from" + str(i) + ".txt.npy")
        y = np.load("C:\\data12\\pic_to" + str(i) + ".txt.npy")
        x = np.expand_dims(np.expand_dims(x, 0),-1)
        y = np.expand_dims(np.expand_dims(y, 0), -1)
        if i == 1999: i = -1
        i += 1
        print(i)
        yield x,y

a = generator()
for i in range(1999):
    #for j in range(20):
    x_data, y_data = next(a)
    n_x_data, n_y_data = next(a)
    show_images([np.reshape(x_data, (ROWS, COLS)), np.reshape(n_x_data, (ROWS, COLS)),
                 np.reshape(x_data, (ROWS, COLS)) - np.reshape(n_x_data, (ROWS, COLS))
                 ], 1, ["1", "2", 'minus'])
