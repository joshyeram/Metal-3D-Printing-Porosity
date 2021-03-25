import csv
from PIL import Image
import numpy as np
import math

value = 1
globalG = 300
def filtering(arr):
    newArr = np.zeros((len(arr) - 2, len(arr) - 2))
    for i in range(1, len(arr) - 1):
        for j in range(1, len(arr) - 1):
            newArr[i - 1][j - 1] = 5 * arr[i][j] - arr[i - 1][j] - arr[i + 1][j] - arr[i][j - 1] - arr[i][j + 1]
    return newArr
def main():
    with open('/Users/joshchung/Desktop/IROriginal/t17p96_x0_y47p30_z1p02_layer3.csv', newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        x = int(data_list[0][1])
        y = int(data_list[1][1])
        print(x)
        print(y)
        threshold = 0
        highest = 0
        mean = 0

        highestj = 0
        highesti = 0

        data = np.zeros((y,x))

        for j in range(0, y):
            for i in range(0, x):
                mean += int(data_list[j+3][i])
                data[j][i] = int(data_list[j+3][i])
                if data[j][i] > float(highest):
                    highest = int(data_list[j+3][i])
                    highestj = j
                    highesti = i
        mean /= y * x
        threshold = int(topPercent(data_list, .01, x, y))

        print(threshold)
        print(mean)
        print(highest)
        print(highestj)
        print(highesti)
        img = Image.new( 'RGB', (x,y), "black")# create a new black image
        pixels = img.load() # create the pixel map
        data1 = np.zeros((40, 40))

        for i in range(x):
            for j in range(y):
                #print(i,j)
                #print(data[j][i])
                if data[j][i] < threshold:
                    continue
                #pixels[i,j] = (sinRGB(int(data_list[j+3][i]), threshold, highest), cosRGB(int(data_list[j+3][i]), threshold, highest), tanRGB(int(data_list[j+3][i]), threshold, highest))
                #pixels[i, j] = (cosRGB(int(data_list[j + 3][i]), threshold, highest), cosRGB1(int(data_list[j + 3][i]), threshold, highest),cosRGB2(int(data_list[j + 3][i]), threshold, highest))
                #pixels[j, i] = (rgb(threshold,highest,data[j][i]))
                print(i,j)
                pixels[i,j] = (red(threshold,highest,data[j][i]),green(threshold,highest,data[j][i]),blue(threshold,highest,data[j][i]))

                #print(data[i][j],red(threshold,highest,data[i][j]),green(threshold,highest,data[i][j]),blue(threshold,highest,data[i][j]))
                ##print(data[i][j],relativePos(threshold, highest, data[i][j]),red(threshold,highest,data[i][j]),green(threshold,highest,data[i][j]),blue(threshold,highest,data[i][j]))
        #pixels[10, 30] = (255,0,0)
        #drawGrad(x,y,highest,threshold,pixels)
        img.show()
        img.save('exampleFullIR.png')
        narr = filtering(data1)
        img1 = Image.new('RGB', (38, 38), "white")  # create a new black image
        pixels1 = img1.load()  #
        for i in range(38):
            for j in range(38):
                #print(i,j)
                #print(data[j][i])
                if narr[j][i] < threshold:
                    continue
                #pixels[i,j] = (sinRGB(int(data_list[j+3][i]), threshold, highest), cosRGB(int(data_list[j+3][i]), threshold, highest), tanRGB(int(data_list[j+3][i]), threshold, highest))
                #pixels[i, j] = (cosRGB(int(data_list[j + 3][i]), threshold, highest), cosRGB1(int(data_list[j + 3][i]), threshold, highest),cosRGB2(int(data_list[j + 3][i]), threshold, highest))
                #pixels[j, i] = (rgb(threshold,highest,data[j][i]))
                pixels1[i,j] = (red(threshold,highest,narr[j][i]),green(threshold,highest,narr[j][i]),blue(threshold,highest,narr[j][i]))
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