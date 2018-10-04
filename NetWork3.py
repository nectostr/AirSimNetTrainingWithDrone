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
# generator -> (X_text, Y_test)

# запилим модель с блекджеком и ...
# когда буду накидывать рекурентные последовательности должны быть stateful
# reset recurrent будет звучать как-то как model.reset_states
path = 'L:\\Documents\\PyCharmProjects\\HelloDrone\\data10'
def generator():
    i = np.random.randint(1,560)
    while True:
        x = np.loadtxt(path + "\\pic_from" + str(i) + ".txt")
        y = np.loadtxt(path + "\\pic_to" + str(i) + ".txt")
        x = np.expand_dims(np.expand_dims(np.expand_dims(x, 0),-1),0)
        y = np.expand_dims(np.expand_dims(y, 0), -1)
        if i == 560: i = 0
        i += 1
        yield x,y
batch_size = 1
#temp_data_x, temp_data_y = next(generator())
#shape_temp_x = temp_data_x.shape
#shape_temp_y = temp_data_y.shape
#print(shape_temp_x, shape_temp_y)

model = Sequential()
K.set_image_data_format("channels_last")
v_max_norm = 2
v_regularizer = 0.0001

model.add(ConvLSTM2D(32, (2, 2), padding='same', activation='relu', stateful=True, return_sequences=True, batch_input_shape=(1,1,ROWS,COLS,1),
                     kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))

model.add(ConvLSTM2D(32, (3, 3), padding='same', activation='relu', stateful=True,
                     kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Reshape((1, 64, 64, 32)))
model.add(ConvLSTM2D(32, (2, 2), padding='same', activation='relu', stateful=True, return_sequences=True,
                     kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))

model.add(ConvLSTM2D(32, (3, 3), padding='same', activation='relu', stateful=True,
                     kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
den_row = int(ROWS / 4)
den_col = int(COLS / 4)
model.add(Dense(den_row * den_col, activation='sigmoid',
                kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
model.add(Dense(den_row * den_col * 2, activation='sigmoid',
                kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
model.add(Dense(den_row * den_col * 16, activation='sigmoid',
                kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
model.add(Reshape((ROWS, COLS , 1)))

opt = keras.optimizers.Adam(lr= 0.001)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])


# data import
print(model.summary())
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
a = generator()
while ep < 4000:

    try:
        print(ep)
        history = model.fit_generator(a, epochs=epochs, steps_per_epoch=10, verbose=1, workers=1)
        if ep % 50 == 0:
            x_data, y_data = next(a)
            res = model.predict(x_data)
            show_images([np.reshape(x_data, (ROWS, COLS)), np.reshape(y_data, (ROWS, COLS)), np.reshape(res,(ROWS, COLS)),
                        ], 1, ["from", "want", "predict"])
        #airsimdata.resetImageConn()
        model.reset_states()
        if ep % 50 ==0:
            model.save('MMmodel123.h5')
    except airsimdata.ExeptInGenData as ex:
        model.reset_states()
    finally:
        ep += 1
print(history.history['loss'])
for i in range(10):
    x_data, y_data = next(generator())
    res = model.predict(x_data)
    show_images([np.reshape(x_data, (ROWS, COLS)), np.reshape(y_data, (ROWS, COLS)), np.reshape(res,(ROWS, COLS)),
                ], 1, ["from", "want", "predict"])
model.save('model.h5')

print("<3")




