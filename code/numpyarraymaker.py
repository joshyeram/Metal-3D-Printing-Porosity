import csv
from PIL import Image
import sys
import glob
import numpy as np

np.set_printoptions(threshold=np.inf)
def main():
    path1 = "/Users/joshchung/Desktop/Sampled1/*.csv"
    paths1 = glob.glob(path1)
    for i in range (len(paths1)):
        name = "/Users/joshchung/Desktop/np4040parted/np"+paths1[i][34:paths1[i].find(".csv")]+"p4.npy"
        test = dataInputCompact(paths1[i])
        print(name)
        np.save(name, test)
"""
1 2
3 4
"""
def dataInputCompact (name):
    with open(name, newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        arr = np.array(data_list).astype(np.float)
        temp = np.zeros((20,20))
        for i in range(20,40):
            for j in range(20,40):
                temp[i-20][j-20] = arr[i][j]
    return np.array(temp)

if __name__ == "__main__":
    main()