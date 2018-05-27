import os
import tempfile
import pprint

import math
from PythonClient.AirSimClient import *
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
    time.sleep(duration)
    collision = client.getCollisionInfo()
    # print(collision)
    if (collision.has_collided == True):
        print("Attention, collision")
        client.moveByVelocityZ(-vx, -vy, z, 2, DrivetrainType.ForwardOnly)
        time.sleep(5)

def moveRight(duration):
    speed = 1
    pitch, roll, yaw  = client.getPitchRollYaw()
    vx = math.cos(yaw + 90) * speed
    vy = math.sin(yaw + 90) * speed
    z = client.getPosition().z_val
    client.moveByVelocityZ(vx, vy, z, duration, DrivetrainType.ForwardOnly)
    time.sleep(duration)
    collision = client.getCollisionInfo()
    # print(collision)
    if (collision.has_collided == True):
        print("Attention, collision")
        client.moveByVelocityZ(-vx, -vy, z, 2, DrivetrainType.ForwardOnly)
        time.sleep(5)


tmp_dir = 'L:\\Pictures\\Uplay'
globalPictureIndex = 0
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

def movementConnection():
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    moveToDir(0, 0, -1)
    print("flied up")
    client.rotateToYaw(0)
    print("rotated")


def flyIteration(n):
    yaw = randint(-90,90)
    client.rotateToYaw(yaw)
    moveRight(n)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def resetImageConn():
    client.reset()
    moveToDir(0, 0, -1)
    print("flied up")
    client.rotateToYaw(0)
    print("rotated")



def processDataForSavingAndForNet():
    difROWs = 2
    difCOLs = 4
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
    # plt.matshow(depth_array)
    # plt.show()
    #---------
    # обработка RGB изображения
    # _________________________
    outX_array = np.reshape(np.fromstring(pics[1].image_data_uint8, dtype=np.uint8) , (pics[1].height, pics[1].width, 4))
    # plt.imshow(outX_array)
    # plt.show()
    outX_array = rgb2gray(outX_array)
    outX_array = outX_array / max(outX_array.flat)
    outX_array = np.expand_dims(np.expand_dims(np.expand_dims(outX_array, 0),0),-1)
    depth_array = np.expand_dims(np.expand_dims(depth_array, 0), -1)
    return (outX_array, depth_array)


def getBatchOfPice(batch_size):
    collision = client.getCollisionInfo()
    if (collision.has_collided == True):
        print("Attention, collision")
        speed = 1
        pitch, roll, yaw = client.getPitchRollYaw()
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        z = client.getPosition().z_val
        client.moveByVelocityZ(-vx, -vy, z, 2, DrivetrainType.ForwardOnly)
        time.sleep(5)
    moveRight(batch_size / 2)
    res = []
    for i in range(batch_size):
        res.append(processDataForSavingAndForNet())
        time.sleep(0.5)
    return res

class ExeptInGenData(Exception):
    pass

def getData():
    f = False
    to_return = ()
    try:
        # flyIteration()
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
        raise ExeptInGenData
    return to_return

connet_ip = "192.168.1.100"
client = MultirotorClient(connet_ip)
client.confirmConnection()
# go up

# a = input()
#print("start")
#moveAhead(15)
#print("end")