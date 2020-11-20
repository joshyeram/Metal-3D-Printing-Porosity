import numpy as np
import csv
from os import path
import glob
import os
import shutil

def main():
    path = "/Users/joshchung/Desktop/sampleBad/bad*.csv"
    data = np.zeros((40, 40))
    for fname in glob.glob(path):
        if(fname.find("90bad")==-1):
            with open(fname, newline='') as csvfile:
                data_list = list(csv.reader(csvfile))
                for i in range(40):
                    for j in range(40):
                        data[i][j] = int(float(data_list[i][j]))
                dataRotated = np.rot90(data, 3)
            print("/Users/joshchung/Desktop/sampleBad/270"+ fname[35:])
            with open("/Users/joshchung/Desktop/sampleBad/270"+ fname[35:], 'w', newline='') as file:
                writer = csv.writer(file)
                for i in range(len(data_list)):
                    writer.writerow(dataRotated[i])


if __name__ == '__main__':
     main()
