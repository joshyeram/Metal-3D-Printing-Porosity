import numpy as np
import csv
from os import path
import glob
import os


def main():

    path = "/Users/joshchung/Desktop/temp/*.csv"
    data = np.zeros((30, 30))
    temp = [0] * 30
    cor = 0
    for fname in glob.glob(path):
        nameNor = fname[30:]
        if(nameNor.find("90bad")!=-1):
            print(nameNor)
            with open(fname, newline='') as csvfile:
                data_list = list(csv.reader(csvfile))
                for i in range(30):
                    for j in range(30):
                        data[i][j] = int(float(data_list[i][j]))
                dataRotated = np.rot90(data, 1)
                print(fname[fname.find("bad")+3:fname.find(".csv")])
                with open("/Users/joshchung/Desktop/temp/180bad"+fname[fname.find("bad")+3:fname.find(".csv")]+".csv", 'w', newline='') as file:
                    writer = csv.writer(file)
                    for i in range(30):  # for every col:
                        for j in range(30):
                            temp[cor] = dataRotated[i][j]
                            cor+=1
                        writer.writerow(temp)
                        cor=0
        """
        if(fname.find("Sampled90")!= -1):
            continue
        if (fname.find("Sampled180") != -1):
            continue
        data = np.zeros((80,80))
        point = fname.find("ad/") + 3
        name = fname[point:]
        with open(fname, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data_list = list(csv.reader(csvfile))
            for i in range(80):
                for j in range(80):
                    data[i][j] = int(float(data_list[i][j]))
            print(data)
            dataRotated = np.rot90(data,3)
            print(dataRotated)
        
        with open('/Users/joshchung/Desktop/sampleBad/' + "Sampled270_" + name[8:], 'w', newline='') as file:
            writer = csv.writer(file)
            temp = [0] * 80
            for i in range(80):  # for every col:
                for j in range(80):
                    temp[j] = dataRotated[i][j]
                writer.writerow(temp)
        with open('porosityName.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            print("Sampled270_"+name[8:])
            writer.writerow(["Sampled270_"+name[8:]])
        """

if __name__ == '__main__':
     main()
