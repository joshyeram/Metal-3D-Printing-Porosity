import csv
import glob
import os.path
from os import path
import shutil
import os
import random

def main():
    crossVal()

    """
    pathi = "/Users/joshchung/Desktop/temp/"
    check = [""] * 1564
    with open('indexData.csv', newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        data = [[0] * 13 for i in range(72)]
    for i in range(1,72):
        for j in range(9):
            #print(i,j)
            data[i-1][j] = int(data_list[i][j])
    for i in range(71):
        check[i] = "Sampled2_t"+str(data[i][1])+"p"+str(data[i][2])+"_x"+str(data[i][3])+"_y"+str(data[i][4])+"p"+str(data[i][5])+"_z"+str(data[i][6])+"p"+str(data[i][7])+"_layer"+str(data[i][8])+".csv"

    for i in range(71):
        #print(i)
        if(path.exists("/Users/joshchung/Desktop/temp/"+check[i]) == True):
            print(check[i],i+1)

    with open('porosityName.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(71):
            # print(i)
            writer.writerow([check[i]])

  
    with open('porosityName.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        path = "/Users/joshchung/Desktop/sampleBad/*.csv"

        for fname in glob.glob(path):
            point = fname.find("ad/") + 3
            name = fname[point:]
            writer.writerow([name])

        for i in range(71):
            writer.writerow([check[i]])
    """

    pathi = "/Users/joshchung/Desktop/Sampled/"
    to = "/Users/joshchung/Desktop/sampleBad/"
    #for i in check:
        #shutil.move(pathi+i,to+i)


def crossVal():
    pathi = "/Users/joshchung/Desktop/temp/*.csv"
    dir = glob.glob(pathi)
    i = 0
    while(i<148):
        rando = random.randint(0,len(dir)-1)
        pathRandom = dir[rando]
        if(pathRandom.find("good")!=-1):
            to = "/Users/joshchung/Desktop/crossVal/group10/"+pathRandom[30:]
            if (path.exists(pathRandom) == True):
                print(to)
                shutil.move(pathRandom,to)
                i+=1
    return


if __name__== "__main__":
   main()

#Sampled_t0p4573_x0_y5p808_z0_layer1