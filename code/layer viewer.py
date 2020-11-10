import csv
from PIL import Image
import numpy as np
import math

value = 1
globalG = 300

def main():
    with open('/Users/joshchung/Desktop/converted/t92p22_x0_y42p59_z5p61_layer12.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(csv.reader(csvfile))
        x = int(data_list[0][1])
        y = int(data_list[1][1])
        print(x)
        print(y)
        threshold = 0
        highest = 0
        mean = 0


        data = [[0]* x for i in range(y)]

        for j in range(0, y):
            for i in range(0, x):
                mean += int(data_list[j+3][i])
                data[j][i] = int(data_list[j+3][i])
                if data[j][i] > float(highest):
                    highest = int(data_list[j+3][i])
        mean /= y * x
        threshold = int(topPercent(data_list, .01, x, y))

        print(threshold)
        print(mean)
        print(highest)

        img = Image.new( 'RGB', (x,y), "black")# create a new black image
        pixels = img.load() # create the pixel map


        for i in range(y):    # for every col:
            for j in range(x):    # For every row
                #print(i,j)
                #print(data[j][i])
                if data[i][j] < threshold:
                    continue
                #pixels[i,j] = (sinRGB(int(data_list[j+3][i]), threshold, highest), cosRGB(int(data_list[j+3][i]), threshold, highest), tanRGB(int(data_list[j+3][i]), threshold, highest))
                #pixels[i, j] = (cosRGB(int(data_list[j + 3][i]), threshold, highest), cosRGB1(int(data_list[j + 3][i]), threshold, highest),cosRGB2(int(data_list[j + 3][i]), threshold, highest))
                #pixels[j, i] = (rgb(threshold,highest,data[j][i]))
                pixels[j,i] = (red(threshold,highest,data[i][j]),green(threshold,highest,data[i][j]),blue(threshold,highest,data[i][j]))
                #print(data[i][j],red(threshold,highest,data[i][j]),green(threshold,highest,data[i][j]),blue(threshold,highest,data[i][j]))
                ##print(data[i][j],relativePos(threshold, highest, data[i][j]),red(threshold,highest,data[i][j]),green(threshold,highest,data[i][j]),blue(threshold,highest,data[i][j]))
        pixels[467, 247] = (0,0,0)
        drawGrad(x,y,highest,threshold,pixels)
        img.show()
"""
def cosRGB(i,t,h):
    return int(globalG / 2) + int(globalG / 2 * math.cos(b(h,t) * (i - t)))
def cosRGB1(i,t,h):
    return int(globalG / 2) + int(globalG / 2 * math.cos(b(h,t) * (i - t) + cosOffset(t,h) ))
def cosRGB2(i, t, h):
    return int(globalG / 2) + int(globalG / 2 * math.cos(b(h,t) * (i - t) +  2 * cosOffset(t,h)))
"""
def cosRGB(i,t,h):
    v =  int(globalG / 2) + int(globalG / 2 * math.cos(b(h,t) * (i - t)))
    if v > 255:
        return 255
    if v < 0:
        return 0
    return v
def cosRGB1(i,t,h):
    v =  int(globalG / 2) + int(globalG / 2 * math.cos(b(h,t) * (i - t) - cosOffset(t,h) ))
    if v > 255:
        return 255
    if v < 0:
        return 0
    return v
def cosRGB2(i, t, h):
    v =  int(globalG / 2) + int(globalG / 2 * math.cos(b(h,t) * (i - t) -  2 * cosOffset(t,h)))
    if v > 255:
        return 255
    if v < 0:
        return 0
    return v
def cosOffset(t, h):
    return ((h-t)/value)/3
def sinRGB(i,t,h):
    return int(255 / 2) + int(255 / 2 * math.sin(b(h,t) * (i - t)))
def tanRGB(i,t,h):
    v = int(255 / 2) + int(255 / 2 * math.tan((2 * b(h,t)) * (i - t)))
    if v > 255:
        return 255
    if v < 0:
        return 0
    return v

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

def b(high, threshold):
    v = 2 * math.pi / ((high-threshold)/ value)
    #print (v)
    #print ((high-threshold)/.7)
    return v

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
            p[j,i] = (red(0,y,i), blue(0,y,i),green(0,y,i))
    return

if __name__ == '__main__':
    main()