import csv
import numpy as np
from PIL import Image, ImageDraw

def dataInput(name):
    data = np.zeros((40,40), dtype=np.float128)
    with open(name, newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        for i in range(40):
            for j in range(40):
                data[i][j] = float(data_list[i][j])
    set = np.array(data)
    return set

"""
case 0: top left
case 1: top
case 2: top right
case 3: left
case 4: right
case 5: bot left
case 6: bot 
case 7: bor right

012
3 4
567

corner    edge    corner
edge               edge
corner    edge    corner
"""
def grad(set):
    data = np.zeros((30, 30), dtype=np.float128)
    for i in range(1,len(set)-1):
        for j in range(1,len(set[0])-1):
            # i , j        0        1         2       3       4         5        6       7
            highest = [[i-1,j-1],[i-1,j],[i-1,j+1],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j],[i+1,j+1]]
            #print(highest)
            data[i][j] = highestPointer(set,highest)
    usableData = np.zeros((28, 28), dtype=np.float128)
    for i in range(28):
        for j in range(28):
            usableData[i][j] = data[i+1][j+1]
    print(usableData)
    return usableData

def vectorGraph(set, pos1, pos2):
    posx = pos1
    posy = pos2
    center = set[posy][posx]
    state = True
    vert = 0.
    hori = 0.
    for i in range (-1, 2):
        for j in range(-1, 2):
            if(center < set[posy + i][posx+j]):
                state = False
                break
    if(state == False):
        for i in range(-1, 2):
            for j in range(-1, 2):
                curr = set[posx + i][posy+j]
                curr = center - curr
                if((i == -1 and j==-1) or (i == -1 and j==1) or (i == 1 and j==-1) or (i == 1 and j==1)):
                    curr /= 1.4142135
                if(i == -1):
                    vert +=curr
                elif(i==1):
                    vert -= curr
                if(j==-1):
                    hori -= curr
                elif(j==1):
                    hori += curr
                if(i==-1 and j ==0):
                    vert += curr
                elif(i==0 and j ==-1):
                    hori -= curr
                elif(i==0 and j ==1):
                    hori += curr
                elif(i==1 and j ==0):
                    vert -= curr
    if(state == False):
        sq = np.sqrt(vert * vert + hori * hori)
    else:
        return (0,0)
    return (hori/sq,vert/sq)

def draw2(usableData):
    store = np.zeros((38,38,2))
    img = Image.new('RGB', (380, 380), "black")  # create a new black image
    img1 = ImageDraw.Draw(img)
    for i in range(1,39):
        for j in range(1,39):
            store[j-1][i-1] = vectorGraph(usableData,j,i)
    scale = 3
    for i in range(0,38):
        for j in range(0,38):
            img1.line([(i * 10 + 5+ scale*store[i][j][1],j * 10 + 5 - scale*store[i][j][0]), (i * 10 + 5-scale*store[i][j][1], j * 10 + 5+scale*store[i][j][0])], width=0)
    img.show()

    return store

def highestPointer(set, highest):
    data = np.zeros(8)
    ptr = 0
    #top left
    data[0] = set[highest[0][0]][highest[0][1]] + set[highest[1][0]][highest[1][1]] + set[highest[3][0]][highest[3][1]]
    #top
    data[1] = set[highest[0][0]][highest[0][1]] + set[highest[1][0]][highest[1][1]] + set[highest[2][0]][highest[2][1]]
    #top right
    data[2] = set[highest[1][0]][highest[1][1]] + set[highest[2][0]][highest[2][1]] + set[highest[4][0]][highest[4][1]]
    #left
    data[3] = set[highest[0][0]][highest[0][1]] + set[highest[5][0]][highest[5][1]] + set[highest[3][0]][highest[3][1]]
    #right
    data[4] = set[highest[2][0]][highest[2][1]] + set[highest[4][0]][highest[4][1]] + set[highest[7][0]][highest[7][1]]
    # bot left
    data[5] = set[highest[3][0]][highest[3][1]] + set[highest[5][0]][highest[5][1]] + set[highest[6][0]][highest[6][1]]
    # bot
    data[6] = set[highest[5][0]][highest[5][1]] + set[highest[6][0]][highest[6][1]] + set[highest[7][0]][highest[7][1]]
    # bot right
    data[7] = set[highest[6][0]][highest[6][1]] + set[highest[7][0]][highest[7][1]] + set[highest[4][0]][highest[4][1]]

    for i in range(8):
        if(data[i]>data[ptr]):
            ptr = i
    return ptr

def draw(usableData):
    img = Image.new('RGB', (280, 280), "black")  # create a new black image
    img1 = ImageDraw.Draw(img)

    for i in range(28):
        for j in range(28):
            if(usableData[i][j]==0 or usableData[i][j]==7):
                img1.line([(j*10,i*10),(j*10+10,i*10+10)], width=0)
            if(usableData[i][j]==2 or usableData[i][j]==5):
                img1.line([(j * 10+10, i * 10), (j * 10, i * 10 + 10)], width=0)
            if (usableData[i][j] == 1 or usableData[i][j] == 6):
                img1.line([(j * 10 + 5, i * 10), (j * 10+5, i * 10 + 10)], width=0)
            if(usableData[i][j] == 3 or usableData[i][j] == 4):
                img1.line([(j * 10, i * 10+5), (j * 10+10, i * 10 + 5)], width=0)
    img.show()

def main():
    pathi = "/Users/joshchung/Desktop/cross/g8/Sampled_t419p8_x0_y38p72_z27p54_layer55.csv"
    set = dataInput(pathi)
    camp = draw2(set)
    for i in range (len(camp)):
        for j in range (len(camp[i])):
            print("(", end="")
            for k in range (2):
                print(camp[i][j][k], end=" ")
            print(")", end="")
            print("", end=" ")
        print()

if __name__ == "__main__":
    main()