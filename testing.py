import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
import keras
from keras.layers import *
from matplotlib import pyplot as plt
import tensorflow as tf
# Highly accurate and scientifically computed parameters
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=11)
np.random.seed(123)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
K.set_session(sess)
ROWS = 64
COLS = 64

# Very complex MSE realization
def mse(first: np.ndarray, second: np.ndarray) -> float:
  return float(np.sum((first - second) ** 2) / len(first.flat))
path = "L:\\Documents\\PyCharmProjects\\HelloDrone\\data10"

def generator():
    i = np.random.randint(1,200)
    while True:
        x = np.loadtxt(path + "\\pic_from" + str(i) + ".txt")
        y = np.loadtxt(path + "\\pic_to" + str(i) + ".txt")
        x = np.expand_dims(np.expand_dims(np.expand_dims(x, 0),-1), 0)
        y = np.expand_dims(np.expand_dims(y, 0), -1)
        if i == 200: i = 0
        i += 1
        yield x,y

model = keras.models.load_model('MMmodel123.h5')
gen = generator()

prev_pic_x, prev_pic_y = next(gen)
prev_pic_res = model.predict(prev_pic_x)
for i in range(0, 100):
    new_pic_x, new_pic_y =next(gen)
    if i % 10 == 0:
        # cv depth map
        cv_prev_pic_x = prev_pic_x * 255
        cv_new_pic_x = new_pic_x * 255
        cv_prev_pic_x = cv_prev_pic_x.astype(np.uint8)
        cv_new_pic_x = cv_new_pic_x.astype(np.uint8)
        cv_depth_map = stereo.compute(cv_prev_pic_x.reshape(64,64),cv_new_pic_x.reshape(64,64))
        cut_large = lambda x: 0 if x < 30 else x
        cut_large = np.vectorize(cut_large)
        cv_depth_map = cut_large(cv_depth_map)

        # network depth map
        net_depth_map = prev_pic_res.reshape(64,64) #!!!!!!

        # real depth map
        real_depth_map = new_pic_y.reshape(64,64)

        # show all images
        fig = plt.figure()
        maps = [cv_prev_pic_x.reshape(64,64), cv_depth_map, net_depth_map.reshape(64,64), real_depth_map]
        titles = ['Image', 'OpenCV depth map', 'Network depth map', 'Real depth map']
        for i in range(4):
          a = fig.add_subplot(1, 4, i + 1)
          a.set_title(titles[i])
          plt.imshow(maps[i], 'gray', label='A')

        # calculate MSEs and print
        print("MSE of OpenCV realization: {}".format(mse(cv_depth_map, real_depth_map)))
        print("MSE of network realization: {}".format(mse(net_depth_map, real_depth_map)))
        print()

        ssim_const_cv = ssim(cv_depth_map / 255, real_depth_map)
        ssim_const_net = ssim(net_depth_map, real_depth_map)
        print("SSIM of OpenCV realization: {}".format(ssim_const_cv))
        print("SSIM of Net realization: {}".format(ssim_const_net))
        plt.show()
    prev_pic_x, prev_pic_y = new_pic_x, new_pic_y
    prev_pic_res = model.predict(prev_pic_x)