import csv
from PIL import Image
import numpy as np

def main():
    img = Image.new('RGB', (30, 30), "black")  # create a new black image
    pixel = img.load()
    with open('/Users/joshchung/Desktop/testCases/180bad61.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(csv.reader(csvfile))
        data = np.zeros((30,30))
        for i in range(30):
            for j in range(30):
                data[i][j] =  float(data_list[i][j])
                pixel[j,i] = (red(1550,1850,data[i][j]),green(1550,1850,data[i][j]),blue(1550,1850,data[i][j]))
    img.show()

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



