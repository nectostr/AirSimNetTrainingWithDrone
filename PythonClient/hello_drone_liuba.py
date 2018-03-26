"""
For connecting to the AirSim drone environment and testing API functionality
"""

import os
import tempfile
import pprint

import math

import time
from AirSimClient import *


# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)

def moveToDir(x_dir, y_dir, z_dir):
    pos = client.getPosition()
    client.moveToPosition(pos.x_val + x_dir, pos.y_val + y_dir, pos.z_val + z_dir, 1, max_wait_seconds = 20)



# state = client.getMultirotorState()
# s = pprint.pformat(state)
# print("state: %s" % s)

# AirSimClientBase.wait_key('Press any key to takeoff')
# client.takeoff(1)
#
# # state = client.getMultirotorState()
# # print("state: %s" % pprint.pformat(state))
try:
    answer = 'r'
    while answer != 'y':
        if answer == 'r':
            client.rotateToYaw(0, max_wait_seconds = 20)
        print('Enter where to move')
        x = float(input())
        y = float(input())
        z = float(input())
        moveToDir(x,y,z)
        print(client.getPosition())
        collision = client.getCollisionInfo()
        # print(collision)
        if (collision.has_collided == True):
            print("Attention, collision")
            moveToDir(-x, -y, -z)
            client.rotateToYaw(0, max_wait_seconds=20)
        print('Finish? y/n')
        answer = input()
except Exception as e:
    print(e)
    client.reset()

# state = client.getMultirotorState()
# print("state: %s" % pprint.pformat(state))
# client.rotateToYaw()
print("coll cope part")
duration = int(input())
speed = 1
pitch, roll, yaw  = client.getPitchRollYaw()
vx = math.cos(yaw) * speed
vy = math.sin(yaw) * speed
z = client.getPosition().z_val
client.moveByVelocityZ(vx, vy, z, duration, DrivetrainType.ForwardOnly)
time.sleep(3)
client.moveByVelocityZ(-vx, -vy, z, duration, DrivetrainType.ForwardOnly)
time.sleep(3)





AirSimClientBase.wait_key('Press any key to take images')
# get camera images from the car
responses = client.simGetImages([
    ImageRequest(0, AirSimImageType.DepthVis),  #depth visualiztion image
    ImageRequest(1, AirSimImageType.Scene), #scene vision image in png format
    ImageRequest(1, AirSimImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
print('Retrieved images: %d' % len(responses))

tmp_dir = "L:\Documents\PyCharmProjects\HelloDrone\images"
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for idx, response in enumerate(responses):

    filename = os.path.join(tmp_dir, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        AirSimClientBase.write_pfm(os.path.normpath(filename + '.pfm'), AirSimClientBase.getPfmArray(response))
    elif response.compress: #png format
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        AirSimClientBase.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: #uncompressed array
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 4) #reshape array to 4 channel image array H X W X 4
        img_rgba = np.flipud(img_rgba) #original image is fliped vertically
        img_rgba[:,:,1:2] = 100 #just for fun add little bit of green in all pixels
        AirSimClientBase.write_png(os.path.normpath(filename + '.greener.png'), img_rgba) #write to png

AirSimClientBase.wait_key('Press any key to reset to original state')
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
