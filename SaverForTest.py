import numpy as np
import time
import PythonClient.airsimWithNet as airsimdata
ROWS = 256
COLS = 256

globalSaveInd = 0
data_save_dir = "C:\\data11"

def saveData(arrays):
    global globalSaveInd
    outx = arrays[0].reshape((ROWS,COLS))
    outy = arrays[1].reshape((ROWS,COLS))
    a = data_save_dir + "\\pic_from" + str(globalSaveInd) + ".txt"
    np.savetxt(a, outx)
    np.savetxt(data_save_dir + "\\pic_to" + str(globalSaveInd) + ".txt", outy)
    globalSaveInd += 1

while(True):
    arra = airsimdata.processDataForSavingAndForNet()
    saveData(arra)
    time.sleep(0.05)