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
        data = np.load('/Users/joshchung/Desktop/np4040/np180badSampled_t0p7622_x0_y9p680_z0_layer1.npy')
        data = filtering(data)
        for i in range(38):
            for j in range(38):
                #data[i][j] =  float(data_list[i][j])
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
    #img1.show()


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

def filtering(arr):
    newArr = np.zeros((len(arr)-2,len(arr)-2))
    for i in range(1,len(arr)-1):
        for j in range(1, len(arr)-1):
            newArr[i-1][j-1] = 5*arr[i][j]-arr[i-1][j]-arr[i+1][j]-arr[i][j-1]-arr[i][j+1]
    print(newArr)
    return newArr
if __name__ == "__main__":
    main()



