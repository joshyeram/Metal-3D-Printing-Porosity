import csv
from PIL import Image
import glob
import numpy as np
import math

value = 1
globalG = 300
rows, cols = (40, 40)

def main():
    path = "/Users/joshchung/Desktop/converted/*.csv"
    count = 0
    for fname in glob.glob(path):
        print(count)
        count +=1
        with open(fname, newline='') as csvfile:
            point = fname.find("ed/") + 3
            name = fname[point:]
            data_list = list(csv.reader(csvfile))
            x = int(data_list[0][1])
            y = int(data_list[1][1])
            convert = np.zeros((y,x))
            final = np.zeros((rows, cols))
            highest = np.array([0,0,0])
            for i in range(0, y):
                for j in range(0, x):
                    #print(i+3,j)
                    convert[i][j] = float(data_list[i+3][j])
                    if convert[i][j] > float(highest[0]):
                        highest[0] = int(data_list[i+3][j])
                        highest[1] = i
                        highest[2] = j
            indexY = int(highest[1] - rows / 2)
            indexX = int(highest[2] - cols / 2)
            for i in range(rows):    # for every col:
                for j in range(cols):    # For every row
                    final[i][j] = convert[i+indexY][j+indexX]
        with open('/Users/joshchung/Desktop/Sampled/Sampled_'+name, 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(rows):    # for every col:
                writer.writerow(final[i])

        
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b
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
def topPercent(l, percent, x, y):
    b = []
    for i in range(len(l)):
        for j in range(len(l[i])):
            b.append(l[i][j])
    b.sort(reverse=True)

    return int(b[int(percent * x * y)])
def drawGrad(x,y,h,t,p):
    for i in range (y):
        for j in range (0,20):
            p[j,i] = (red(0,y,i),green(0,y,i),blue(0,y,i))
    return
def draw(cols,rows,threshold, highest):
    img = Image.new('RGB', (cols, rows), "black")  # create a new black image
    pixels = img.load()  # create the pixel map

    for i in range(cols):  # for every col:
        for j in range(rows):  # For every row
            if sample[i][j] < threshold:
                continue
            pixels[j, i] = (red(threshold, highest, sample[i][j]), green(threshold, highest, sample[i][j]),
                            blue(threshold, highest, sample[i][j]))
    # pixels[50, 50] = (155,155,155)
    img.show()

if __name__ == '__main__':
    main()