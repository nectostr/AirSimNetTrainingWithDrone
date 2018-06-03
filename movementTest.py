import os
import tempfile
import pprint

import math
from PythonClient.AirSimClient import *
from random import randint
import time
import numpy as np
from matplotlib import pyplot as plt
ROWS = 64
COLS = 64

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

def newmoveAhead(duration):
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

globalSaveInd = 0
data_save_dir = "L:\\Documents\\PyCharmProjects\\HelloDrone\\data"
def saveData(arrays):
    global globalSaveInd
    outx = arrays[0].reshape((ROWS,COLS))
    outy = arrays[1].reshape((ROWS,COLS))
    a = data_save_dir + "\\pic_from" + str(globalSaveInd) + ".txt"
    np.savetxt(a, outx)
    np.savetxt(data_save_dir + "\\pic_to" + str(globalSaveInd) + ".txt", outy)
    globalSaveInd += 1


def movementConnection():
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    moveToDir(0, 0, -1)
    print("flied up")

def moveRight(duration):
    speed = 1
    pitch, roll, yaw  = client.getPitchRollYaw()
    angle = 90 * math.pi / 180
    vx = math.cos(yaw + angle) * speed
    vy = math.sin(yaw + angle) * speed
    z = client.getPosition().z_val
    client.moveByVelocityZ(vx, vy, z, duration, DrivetrainType.ForwardOnly)
    time.sleep(duration)

def moveLeft(duration):
    speed = 1
    pitch, roll, yaw  = client.getPitchRollYaw()
    angle = 90 * math.pi / 180
    vx = math.cos(yaw - angle) * speed
    vy = math.sin(yaw - angle) * speed
    z = client.getPosition().z_val
    client.moveByVelocityZ(vx, vy, z, duration, DrivetrainType.ForwardOnly)
    time.sleep(duration)

def flyIteration(n):
    pitch, roll, yaw = client.getPitchRollYaw()
    Nyaw = randint(-90,90)
    yaw = yaw * 180 / math.pi
    client.rotateToYaw(yaw + Nyaw)
    moveAhead(n)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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
    # plt.matshow(depth_array)
    # plt.show()
    #---------
    # обработка RGB изображения
    # _________________________
    outX_array = np.reshape(np.fromstring(pics[1].image_data_uint8, dtype=np.uint8) , (pics[1].height, pics[1].width, 4))
    # plt.imshow(outX_array)
    # plt.show()
    outX_array = rgb2gray(outX_array)
    outX_array = np.expand_dims(np.expand_dims(np.expand_dims(outX_array, 0),0),-1)
    depth_array = np.expand_dims(depth_array, 0)
    return (outX_array, depth_array)


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

connet_ip = ""
client = MultirotorClient(connet_ip)
client.confirmConnection()

movementConnection()
client.moveToPosition(-11, 12, -1, 1)
client.rotateToYaw(-0.6111164247 * 180 / math.pi)
# a = []
# c = input()
# while c != '`':
#     if (c == 'w'):
#         moveAhead(5)
#     elif (c == 'a'):
#         moveLeft(5)
#     elif c == 'd':
#         moveRight(5)
#     elif c == 'q':
#         pitch, roll, yaw = client.getPitchRollYaw()
#         print(yaw, yaw + 90 * math.pi / 180)
#         yawG = yaw * 180 / math.pi
#         client.rotateToYaw(yawG - 30)
#         print(yaw)
#     elif c == 'e':
#         pitch, roll, yaw = client.getPitchRollYaw()
#         print(yaw, yaw + 90 * math.pi / 180)
#         yawG = yaw * 180 / math.pi
#         client.rotateToYaw(yawG + 30)
#         print(yaw)
#     position = client.getPosition()
#     print(position.x_val, position.y_val, position.z_val)
#     pics = processDataForSavingAndForNet()
#     #plt.matshow(pics[0].reshape(64,64))
#     plt.matshow(pics[1].reshape(64, 64))
#     plt.show()
#     c = input()
#     a.append(c)
# print(a)
# circle iteration
time.sleep(1)
angle_to_turn = 17
while (True):
    for i in range(int(360 / angle_to_turn) * 100):
        print(i)
        moveRight(21)
        pitch, roll, yaw = client.getPitchRollYaw()
        yawG = yaw * 180 / math.pi
        angle_del = (yawG - angle_to_turn)
        if angle_del < -180:
            angle_del = 360 + angle_del
        elif angle_del > 180:
            angle_del = angle_del - 360
        client.rotateToYaw(angle_del)
        #pics = processDataForSavingAndForNet()
        # plt.matshow(pics[0].reshape(64,64))
        #plt.matshow(pics[1].reshape(64, 64))
        #plt.show()

# reading data
# for i in range(0, 560, 30):
#     x = np.loadtxt("L:\\Documents\\PyCharmProjects\\HelloDrone\\data\\pic_from"+str(i)+".txt")
#     y = np.loadtxt("L:\\Documents\\PyCharmProjects\\HelloDrone\\data\\pic_to"+str(i)+".txt")
#     plt.matshow(x)
#     plt.matshow(y)
# plt.show()

# заменить подъезл на мове то(хуз) и поворт автоматически. ЗАмерить сколько именно над линеей и детать наиней.
#  Попробовать подсчитать как говорил рома
# -10.407453536987305 11.082700729370117
# -13.337678909301758 13.63478946685791
# -6.424674231358698e-06
# -0.7983215994267586


