import os
import tempfile
import pprint

import math
from AirSimClient import *
from random import randint
import time
import numpy as np
from matplotlib import pyplot as plt

def moveToDir(x_dir, y_dir, z_dir):
    pos = client.getPosition()
    client.moveToPosition(pos.x_val + x_dir, pos.y_val + y_dir, pos.z_val + z_dir, 1) # , max_wait_seconds = 20

def moveAhead(duration):
    speed = 1
    pitch, roll, yaw  = client.getPitchRollYaw()
    vx = math.cos(yaw) * speed
    vy = math.sin(yaw) * speed
    z = client.getPosition().z_val
    client.moveByVelocityZ(vx, vy, z, duration, DrivetrainType.ForwardOnly)
    time.sleep(3)
    collision = client.getCollisionInfo()
    # print(collision)
    if (collision.has_collided == True):
        print("Attention, collision")
        client.moveByVelocityZ(-vx, -vy, z, 2, DrivetrainType.ForwardOnly)
        time.sleep(5)


globalPictureIndex = 0
tmp_dir = "L:\\Documents\\PyCharmProjects\\HelloDrone\\testData"
def savePictures(responses):
    global globalPictureIndex
    for idx, response in enumerate(responses):
        filename = os.path.join(tmp_dir,  "pic_" + str(globalPictureIndex) + "_" + str(idx))
        globalPictureIndex += 1
        if response.compress:  # png format
            AirSimClientBase.write_file(os.path.normpath(filename + '.JPG'), response.image_data_uint8)
        else:  # uncompressed array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgba = img1d.reshape(response.height, response.width, 4)  # reshape array to 4 channel image array H X W X 4
            img_rgba = np.flipud(img_rgba)  # original image is fliped vertically
            AirSimClientBase.write_png(os.path.normpath(filename + '.png'), img_rgba)  # write to png



def takePictures():
    responses = client.simGetImages([
        ImageRequest(0, AirSimImageType.DepthPerspective, True, False),  # depth visualiztion image
        ImageRequest(0, AirSimImageType.Scene, False, False)   # scene vision image in png format
    ])
    return responses


# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)

def flyIteration():
    yaw = randint(-90,90)
    print(yaw)
    client.rotateToYaw(yaw)
    moveAhead(2)


def processDataForSavingAndForNet():
    pics = takePictures()
    # print(pics[0])
    #-------
    # Обработка карты глубины
    depth_array = np.reshape(np.asarray(pics[0].image_data_float, dtype=np.float32), (pics[0].height, pics[0].width))
    # ограничим дальность карты глубины до x
    def dep_lim(x):
        return x if x < 50 else 50
    dep_lim = np.vectorize(dep_lim)
    # Срезаем до 50м карту глубины
    depth_array = dep_lim(depth_array)
    # нормализуем до 0-1
    depth_array = depth_array / max(depth_array.flat)
    # визуализация для себя
    plt.matshow(depth_array)
    plt.show()

    #---------
    # обработка RGB изображения
    # _________________________
    RGB_array = np.reshape(np.fromstring(pics[1].image_data_uint8, dtype=np.uint8) , (pics[1].height, pics[1].width, 4))
    plt.imshow(RGB_array)
    plt.show()
    print(depth_array.shape)
    print(RGB_array.shape)
    return (RGB_array, depth_array)

x_min = -10
x_max = 36
y_min = -17
y_max = 14

# go up
moveToDir(0,0,-1)
print("flied up")
client.rotateToYaw(0)
print("rotated")
# processDataForSavingAndForNet()
# a = input()

def getData():
    f = False
    to_return = ()
    try:
        flyIteration()
        pos = client.getPosition()
        to_return = processDataForSavingAndForNet()
    # если он вышел за границу области или была огшибка
    except:
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        moveToDir(0, 0, -1)
        print("flied up")
        client.rotateToYaw(0)
        print("rotated")
        f = True
    return (to_return, f)

# for i in range(10):
#     flyIteration()
#     print("fly iteration" +  str(i))



# client.reset()
