import csv
from PIL import Image
import sys
import glob
import numpy as np

np.set_printoptions(threshold=np.inf)
def main():
    path1 = "/Users/joshchung/Desktop/Sampled/*.csv"
    paths1 = glob.glob(path1)
    for i in range (len(paths1)):
        name = "/Users/joshchung/Desktop/nparrays/np"+paths1[i][33:paths1[i].find(".csv")]+".npy"
        test = dataInputCompact(paths1[i])
        print(name)
        np.save(name, test)

def dataInputCompact (name):
    with open(name, newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        arr = np.array(data_list).astype(np.float)
    return np.array(arr)

if __name__ == "__main__":
    main()