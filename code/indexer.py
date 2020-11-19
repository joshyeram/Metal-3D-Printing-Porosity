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
    pathi = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/visual/porosityName.csv"
    check = [""] * 1564
    with open(pathi, newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
    
    data = [[0] * 13 for i in range(72)]
    for i in range(1,72):
        for j in range(9):
            #print(i,j)
            data[i-1][j] = int(data_list[i][j])
    
    for i in range(71):
        check[i] = "Sampled_t"+str(data_list[i][1])+"p"+str(data_list[i][2])+"_x"+str(data_list[i][3])+"_y"+str(data_list[i][4])+"p"+str(data_list[i][5])+"_z"+str(data_list[i][6])+"p"+str(data_list[i][7])+"_layer"+str(data_list[i][8])+".csv"

    for i in range(71):
        if(path.exists("/Users/joshchung/Desktop/Sampled/"+check[i]) == True):
            print(check[i],i+1)

    with open('porosityName.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        patho = "/Users/joshchung/Desktop/sampleBad/*.csv"

        for fname in glob.glob(patho):
            point = fname.find("ad/") + 3
            name = fname[point:]
            writer.writerow([name])

        for i in range(71):
            writer.writerow([check[i]])
    

    pathi = "/Users/joshchung/Desktop/Sampled/"
    to = "/Users/joshchung/Desktop/sampleBad/"
    #for i in check:
        #shutil.move(pathi+i,to+i)
    """

def crossVal():
    pathi = "/Users/joshchung/Desktop/sampleBad/*.csv"
    for fname in glob.glob(pathi):
        with open("/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/visual/porosityName.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([fname[35:]])

if __name__== "__main__":
   main()

#Sampled_t0p4573_x0_y5p808_z0_layer1