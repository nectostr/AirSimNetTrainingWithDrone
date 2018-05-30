import struct
from PIL import Image
import numpy as np

def bin2float(filepath: str, length: int) -> tuple:
    """
    read (length) floats from raw byte file and return tuple of them

    :param filepath: path to binary file
    :param length: number of floats inside (each float is 32-bit number)
    :return: tuple with floats
    """
    with open(filepath, 'rb') as f:
        ans = struct.unpack(str(length) + "f", f.read())

    return ans


def xcr_reader(filepath: str, col_num: int, row_num: int, offset: int = 0, reversed: bool = False) -> (int, int, int, np.ndarray):
    """
    read xcr image

    :param filepath: path to xcr file
    :param col_num: number of columns in img
    :param row_num: number of rows in img
    :param offset: offset from begin (for header)
    :param reversed: if bytes are reversed in image
    :return: (columns_number, rows_number, depth, image_array)
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    image_len = col_num * row_num

    depth = 16

    image_data = list(data[offset:(offset + image_len*2)])

    if reversed:
        for i in range(0, len(image_data), 2):
            image_data[i], image_data[i+1] = image_data[i+1], image_data[i]

    image_data = list(struct.unpack(str(image_len) + "h", bytes(image_data)))
    image_data = np.array(image_data).reshape((col_num, row_num))

    image_data = np.flipud(image_data)
    return col_num, row_num, depth, image_data


def jpg_reader(filepath: str, channels: int = 3) -> (int, int, int, np.ndarray):
    """
    read jpg files

    :param filepath: path to jpg file
    :return: (columns_number, rows_number, depth, image_array)
    """
    assert channels == 1 or channels == 3, "channels number must be 1 or 3"

    jpgfile = Image.open(filepath)

    col_num = jpgfile.width
    row_num = jpgfile.height

    depth = jpgfile.bits

    image_data = np.array(jpgfile.getdata(), dtype=np.ubyte)

    cur_channels = len(image_data.flat) / row_num / col_num
    if cur_channels // 1 != cur_channels:
        raise ValueError("inconsistent data: can't divide into channels")
    cur_channels = int(cur_channels)

    # todo: refactor
    if channels == 3:
        if cur_channels == 1:
            image_data = image_data.reshape((row_num, col_num))
            image_data = np.stack((image_data,) * 3, -1)
        else:
            image_data = image_data.reshape((row_num, col_num, cur_channels))
    else:
        if cur_channels != 1:
            image_data = image_data.reshape((row_num, col_num, cur_channels))
            image_data2 = np.empty(shape=(image_data.shape[0], image_data.shape[1]))
            for i in range(len(image_data)):
                image_data2[i, :] = image_data[i, :, 0]
            image_data = image_data2
        else:
            image_data = image_data.reshape((row_num, col_num))

    return col_num, row_num, depth, image_data

col_num, row_num, depth, image_data = jpg_reader("stones.jpg",1)


import matplotlib.pyplot as plt
plt.matshow(image_data, cmap="gray")

mid_in_neibr = np.ndarray(shape = image_data.shape, dtype = image_data.dtype)

image_data = np.exp(image_data/image_data.std())

for i in range(1, row_num - 2):
    for j in range(1, col_num - 2):
        tmp = int((image_data[i + 1, j + 1] + image_data[i+1,j] + image_data[i+1, j-1] +
        image_data[i, j+1] + image_data[i, j - 1] +
        image_data[i-1, j+ 1] + image_data[i-1, j] + image_data[i-1, j-1]) / 8)
        mid_in_neibr[i,j] = tmp



print("done")

res = mid_in_neibr + mid_in_neibr / 2

for i in range(1, row_num - 2):
    for j in range(1, col_num - 2):
        if res[i,j] > 12: res[i,j] = 100
        else: res[i,j] = 0

watered = np.ndarray(shape = image_data.shape, dtype = image_data.dtype)
need_to_water = np.ndarray(shape = image_data.shape, dtype = image_data.dtype)
need_to_water[1,1] = 200
for i in range(1, row_num-1):
    for j in range(1, col_num-1):
        if (image_data[i,j] < 140 and image_data[i,j] <=  need_to_water[i,j] + 5 ):
            watered[i,j] = 255
            mid = image_data[i,j]
            need_to_water[i + 1, j] = mid
            need_to_water[i - 1, j] = mid
            need_to_water[i, j + 1] = mid
            need_to_water[i, j - 1] = mid
        else:
            watered[i,j] = 0



plt.matshow(watered)
plt.matshow(need_to_water)
plt.matshow(watered - image_data)
plt.show()


"""for i in range(row_num):
    for j in range(col_num):
        if (image_data[i,j] > 120): image_data[i,j] = 255
        else: image_data[i,j] = 0

for i in range(1, row_num-1):
    for j in range(1,col_num-1):
        if ((image_data[i,j] == 0) and
                ((image_data[i-1, j-1] + image_data[i - 1, j] + image_data[i - 1, j + 1]+
                image_data[i, j - 1] + image_data[i, j+ 1] +
                image_data[i + 1, j - 1] + image_data[i+1, j] + image_data[i+1, j + 1])/8 > 191)):
            image_data[i,j] = 255
plt.imshow(image_data, cmap="gray")
plt.show()"""

"""
amount_of_stones = 0
amount_of_pixels = 7
mid_more = 150
mid_less = 160
smalest_stone = 130
i = 0
while(i < col_num - amount_of_pixels):
    j = 0
    while (j < row_num - amount_of_pixels):
        sum_right = 0
        sum_down = 0
        sum_diag = 0
        min_right = 255
        min_down = 255
        min_diag = 255
        for k in range(amount_of_pixels):
            sum_right += image_data[j, i + k]
            if (min_right > image_data[j, i + k]): min_right = image_data[j, i + k]
            sum_down += image_data[j + k, i]
            if (min_down > image_data[j + k, i]): min_down = image_data[j + k, i]
            sum_diag += image_data[j + k, i + k]
            if (min_diag > image_data[j + k, i + k]): min_diag = image_data[j + k, i + k]
        sum_right /= amount_of_pixels
        sum_down /= amount_of_pixels
        sum_diag /= amount_of_pixels
        if (sum_diag > mid_more and sum_diag < mid_less and min_diag > smalest_stone) or \
                (sum_right > mid_more and sum_right < mid_less and min_right > smalest_stone) or \
                (sum_down > mid_more and sum_down < mid_less and min_down < smalest_stone):
            amount_of_stones += 1
            for k in range(amount_of_pixels):
                image_data[j, i + k] = 100
                image_data[j + k, i] = 100
                image_data[j + k, i + k] = 100
            j += 9
        else:
            j += 1
    i+=1
"""
