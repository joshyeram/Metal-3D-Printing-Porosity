import csv
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import glob
import numpy as np
import shutil

def main():
    sizeX = 1700
    sizeY = 1700
    img = Image.new('RGB', (sizeX, sizeY), "black")  # create a new black image
    pixel = img.load()
    x = y = 0
    loco = 0
    threshold = 1525
    highest = 1750
    low = 1800
    data = [[0] * x for i in range(y)]
    path = "/Users/joshchung/Desktop/np4040parted/*.npy"
    for fname in glob.glob(path):
        combine = 0
        data = np.load(fname)
        for j in range(0, 20):
            for i in range(0, 20):
                #data[j][i] = int(float(data_list[j][i]))
                if(data[j][i]>highest):
                    highest = data[j][i]
                elif(data[j][i]<low and data[j][i]!=0):
                    low = data[j][i]
                if(data[j][i]<threshold):
                    continue
                pixel[i+x,j+y] = (red(threshold, highest, data[j][i]), green(threshold, highest, data[j][i]), blue(threshold, highest, data[j][i]))
                combine +=1
        if(combine<300):
            #print(fname[30:])
            combine =1
            #shutil.move(fname,"/Users/joshchung/Desktop/sampleBad"+ fname[32:])
        x=x+20
        if(x>=sizeX):
            x = 0
            y = y + 20
    #drawGrad(sizeX,sizeY,0,0,pixel)
    font = ImageFont.truetype('/Users/joshchung/Downloads/open-sans/OpenSans-Light.ttf', 60)
    text = 'This is a collection of the all the heat maps of the melt pool in the 2016 study conducted at Mississippi State University.\nThere are around 1565 individual melt pool heat maps represented in this image, 80 * 80 pixels resized at (467, 247)\nper melt pool, and each are colored with a temperature gradient starting from around 1550 to 1850 Fahrenheit. The\nline gradient on the left represents the relative temperature to all the heat maps, violet being the hottest temperature\n (~1850) and red being the threshold temperature (~1550).'
    #ImageDraw.Draw(img).text((0, 3200),text,(255, 255, 255),font)
    img.show()
    print(highest, low)
    img.save("4040partedrot3.jpg")


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
def drawGrad(x,y,h,t,p):
    for i in range (1400):
        for j in range (0,5):
            p[j,y-i-1] = (red(0,y,i), green(0,y,i),blue(0,y,i))
    return

if __name__ == '__main__':
    main()