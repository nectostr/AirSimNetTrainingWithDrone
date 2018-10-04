import os
import tempfile
import pprint

import math
from PythonClient.AirSimClient import *
from random import randint
import time
import numpy as np
from matplotlib import pyplot as plt
ROWS = 256
COLS = 256
from PythonClient.airsimWithNet import *

client.moveToPosition(-11, 12, -1, 1)
client.rotateToYaw(-0.6111164247 * 180 / math.pi)

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


