import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
from keras.regularizers import *
from keras.constraints import *
from keras.models import Sequential
from keras.layers import *
from matplotlib import pyplot as plt

def generateNetwork(ROWS, COLS):
    model = Sequential()
    K.set_image_data_format("channels_last")
    v_max_norm = 2
    v_regularizer = 0.0001
    model.add(Conv2D(32, (2, 2), padding='same', activation='relu', batch_input_shape=(1, ROWS, COLS, 1),
                     kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    model.add(Conv2D(32, (2, 2), padding='same', activation='relu',
                     kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Reshape((1, 128, 128, 32)))
    model.add(ConvLSTM2D(16, (2, 2), padding='same', activation='relu', return_sequences=True,
                         kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    model.add(ConvLSTM2D(16, (2, 2), padding='same', activation='relu',
                         kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Reshape((1, 64, 64, 16)))
    model.add(ConvLSTM2D(16, (2, 2), padding='same', activation='relu', stateful=True, return_sequences=True,
                         kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))

    model.add(ConvLSTM2D(16, (3, 3), padding='same', activation='relu', stateful=True,
                         kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.15))
    model.add(Reshape((1, 32, 32, 16)))
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
    # model.add(Dense(den_row * den_col * 2, activation='sigmoid',
    #                kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    # model.add(Dense(den_row * den_col * 2, activation='sigmoid',
    #                kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    model.add(Dense(den_row * den_col * 16, activation='sigmoid',
                    kernel_regularizer=l2(v_regularizer), kernel_constraint=max_norm(v_max_norm)))
    model.add(Reshape((ROWS, COLS, 1)))

    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mean_absolute_error',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
