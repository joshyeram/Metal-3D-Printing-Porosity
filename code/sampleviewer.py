import csv
from PIL import Image
import numpy as np

def main():
    img = Image.new('RGB', (40, 40), "black")  # create a new black image
    pixel = img.load()
    with open('/Users/joshchung/Desktop/4040testcases/90badSampled_t1p067_x0_y13p55_z0_layer1.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(csv.reader(csvfile))
        x = len(data_list)
        data = np.zeros((x,x))
        for i in range(x):
            for j in range(x):
                data[i][j] =  float(data_list[i][j])
                pixel[j,i] = (red(1550,1850,data[i][j]),green(1550,1850,data[i][j]),blue(1550,1850,data[i][j]))
    img.show()

    img1 = Image.new('RGB', (40, 40), "black")  # create a new black image
    pixel1 = img1.load()
    with open('/Users/joshchung/Desktop/4040testcases/90badSampled_t1p067_x0_y13p55_z0_layer1.csv',newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        arr = np.array(data_list).astype(np.float)
        high = np.amax(arr)
        low = np.amin(arr)
        arr -= low
        arr /= (high - low)

    for i in range(40):
        for j in range(40):
            pixel1[j, i] = (int(255 * arr[i][j]), int(255 * arr[i][j]), int(255 * arr[i][j]))
    img1.show()
def red(min,max,val):
    if (relativePos(min, max, val) == 0):
        return 255
    elif (relativePos(min, max, val) == 1):
        return int(-255/.2 * ((val-min)/(max-min)-.2)+255)
    elif (relativePos(min, max, val) == 2):
        return 0
    elif (relativePos(min, max, val) == 3):
        return 0
    else:
        return int(255/.2 * ((val-min)/(max-min)-.8))
def green(min,max,val):
    if (relativePos(min, max, val) == 0):
        return int(255/.2 * ((val-min)/(max-min)))
    elif (relativePos(min, max, val) == 1):
        return 255
    elif (relativePos(min, max, val) == 2):
        return 255
    elif (relativePos(min, max, val) == 3):
        return int(-255/.2 * ((val-min)/(max-min)-.6)+255)
    else:
        return 0
def blue(min,max,val):
    if (relativePos(min, max, val) == 0):
        return 0
    elif (relativePos(min, max, val) == 1):
        return 0
    elif (relativePos(min, max, val) == 2):
        return int(255/.2 * ((val-min)/(max-min)-.4))
    elif (relativePos(min, max, val) == 3):
        return 255
    else:
        return 255
def relativePos(min, max, val):
    if((val-min)/(max-min)>.8):
        return 4
    elif ((val-min)/(max-min) > .6):
        return 3
    elif ((val-min)/(max-min) > .4):
        return 2
    elif ((val-min)/(max-min) > .2):
        return 1
    else:
        return 0
if __name__ == "__main__":
    main()



