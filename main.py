# net import
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
from keras.regularizers import *
from keras.constraints import *
from keras.models import Sequential
from keras.layers import *
from matplotlib import pyplot as plt
import sys
import generateNetwork as gn
np.random.seed(123)

parmas = sys.argv
if 'save' in parmas:
    pass

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
K.set_session(sess)

def fun():
    pass

def show_images(images, cols=1, titles=None):
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

def generatorOffline(path, amount):
    i = np.random.randint(1,amount)
    while True:
        x = np.loadtxt(path + "\\pic_from" + str(i) + ".txt")
        y = np.loadtxt(path + "\\pic_to" + str(i) + ".txt")
        x = np.expand_dims(np.expand_dims(x, 0),-1)
        y = np.expand_dims(np.expand_dims(y, 0), -1)
        if i == amount: i = 0
        i += 1
        yield x,y

def generatorOfflineBin(path, amount):
    i = 0
    while True:
        x = np.load(path + "\\pic_from" + str(i) + ".txt.npy")
        y = np.load(path + "\\pic_to" + str(i) + ".txt.npy")
        x = np.expand_dims(np.expand_dims(x, 0), -1)
        y = np.expand_dims(np.expand_dims(y, 0), -1)
        if i == amount: i = -1
        i += 1
        yield x, y

def generatorOnline():
    import PythonClient.airsimWithNet as airsimdata
    while True:
      yield airsimdata.processDataForSavingAndForNet()

ROWS = 256
COLS = 256
load_path = "C:\\data11"
load_amount = 1999

param1 = 'generate' # load
param2 = 'offlineBin' # online / offlineBin
param3 = 'train' # test

if param1 == 'generate':
    model = gn.generateNetwork(ROWS, COLS)
else:
    model = keras.models.load_model('model50.h5')

if param2 == 'offline':
    generator = generatorOffline(load_path, load_amount)
elif param2 == 'offlineBin':
    generator = generatorOfflineBin(load_path, load_amount)
else:
    generator = generatorOnline()

print(model.summary())
keras.utils.plot_model(model, "L:\\Documents\PyCharmProjects\\HelloDrone\\modelplot.png", True)

if param3 == 'train':
    ep = 0
    epochs = 1
    while ep < 400:
        try:
            print(ep)
            history = model.fit_generator(generator, epochs=epochs, steps_per_epoch=100, verbose=1, workers=1)
            if ep % 10 == 0:
                x_data, y_data = next(generator)
                res = model.predict(x_data)
                show_images(
                    [np.reshape(x_data, (ROWS, COLS)), np.reshape(y_data, (ROWS, COLS)), np.reshape(res, (ROWS, COLS)),
                     ], 1, ["from", "want", "predict"])
            # airsimdata.resetImageConn()
            model.reset_states()
            if ep % 5 == 0:
                model.save('model_data11.h5')
        except airsimdata.ExeptInGenData as ex:
            model.reset_states()
        except Exception as e:
            print(e)
            raise e
        finally:
            ep += 1
else:
    for i in range(10):
        x_data, y_data = next(generator)
        res = model.predict(x_data)
        show_images([np.reshape(x_data, (ROWS, COLS)), np.reshape(y_data, (ROWS, COLS)), np.reshape(res, (ROWS, COLS)),
                     ], 1, ["from", "want", "predict"])