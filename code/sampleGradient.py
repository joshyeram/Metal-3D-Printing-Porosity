import csv
import numpy as np
from PIL import Image, ImageDraw

def dataInput(name):
    data = np.zeros((30,30), dtype=np.float128)
    count = 0
    high = 0
    low = 1500
    with open(name, newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        for i in range(30):
            for j in range(30):
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
    pathi = "/Users/joshchung/Desktop/crossVal/group1/good855.csv"
    set = dataInput(pathi)
    draw(grad(set))

if __name__ == "__main__":
    main()