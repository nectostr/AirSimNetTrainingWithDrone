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
np.random.seed(123)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
K.set_session(sess)
ROWS = 256
COLS = 256