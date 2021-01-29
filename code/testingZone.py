import numpy as np
from numpy import random
import glob
import csv
data = [1,0,1,0]
weights = [1,2,5,4]


weights2 = np.ones((4, 1))
def mult(x,y):
    set = np.zeros((len(x),len(y)))
    lenX = len(x)
    lenY = len(y)
    for i in range(lenX):
        for j in range(lenY):
            set[i][j] = x[i]*y[j]
    return set


def main():
    bias1 = np.random.rand(1,1)
    print(bias1)


def dataInput():
    with open("/Users/joshchung/Desktop/testCases/90bad10.csv", newline='') as csvfile:
        data_list = list(csv.reader(csvfile))

        print(data_list)
        print(float(data_list[0][0]))
        arr = np.array(data_list)
        y = arr.astype(np.float).flatten()
    a = np.rot90(y).flatten()
    print(a)



def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

if __name__ == "__main__":
    main()
    """
    layer1 = np.zeros(hiddenNodes)
    output = np.zeros(1)
    for i in range(hiddenNodes):
        layer1[i] = calc(data,weights1[i])

    output = 0;
    for i in range(hiddenNodes):
        output = output + layer1[i]*weights2[i]
    print(len(weights1))
    print(len(weights2))
    print(output)
    output = sigmoid(output)
    print(output)
    
    
    
        path = "/Users/joshchung/PycharmProjects/Arestry/*.csv"
    for fname in glob.glob(path):
        point = fname.find("Arestry/") + 8
        print(fname[point:])

    """