import csv
import numpy as np
import os.path

"""
@author: Joshua Chung (jyc70)
Project: Rutgers Aresty Research Assistant 2020 - 2021

cnn layout: 
input: 40x40 matrix input -> filter (sharpen filter 5-1-1-1-1) -> Input transformed (38 x 38) -> flatten -(weight1) layer1 -(weight2)> layer 2 -(weight3)> output 
"""

def filtering(arr):
    newArr = np.zeros((len(arr) - 2, len(arr) - 2))
    for i in range(1, len(arr) - 1):
        for j in range(1, len(arr) - 1):
            newArr[i - 1][j - 1] = 5 * arr[i][j] - arr[i - 1][j] - arr[i + 1][j] - arr[i][j - 1] - arr[i][j + 1]
    div = np.amax(newArr) - np.amin(newArr)
    newArr -= np.amin(newArr)
    newArr /= div
    return np.array([newArr.flatten()])

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

"""
def test(dir,state)

dir: directory of temperature data
state: If true, print good or bad. Else, do not print good or bad

1. Checks if dir is valid
2.  a) if combined.npy weight file is found, load weights
    b) else if combined.csv weight file is found, load weight
    c) return false if no weight files are found
3.  a) dir file is .csv
        aa) get 4040 matrix if it has row and col numbers (formatted)
        ab) get 4040 matrix if it does not have row and col numbers (non-formatted or raw)
        ac) get 4040 matrix if file size is 4040
        ad) return false for not valid .csv file
    b) dir file is .npy
        ba) assuming that the file is sampled to 4040, load up .npy
        bb) if it is not sampled, get the 4040 matrix from npy temperature map
    c) return if not .cvs or .npy
4.  if state is true, print out good or bad
    return the guess value from cnn 
    
For devs: you should have the weights as global instead of loading weights everytime you call this method
"""

def test(dir,state):
    if os.path.isfile(dir) == False:
        print("No such file found")
        return False
    if os.path.isfile('combined.npy'):
        wArr = np.load('combined.npy', allow_pickle=True)
        weight1 = wArr[0]
        weight2 = wArr[1]
        weight3 = wArr[2]
    elif os.path.isfile('combined.csv'):
        inputSize = 1444
        hiddenSize1 = 75
        hiddenSize2 = 25
        outputSize = 1
        with open('combined.csv', newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
        weight1 = np.zeros((inputSize, hiddenSize1))
        weight2 = np.zeros((hiddenSize1, hiddenSize2))
        weight3 = np.zeros((hiddenSize2, outputSize))
        for i in range(inputSize):
            for j in range(hiddenSize1):
                weight1[i][j] = float(data_list[i][j])
        for i in range(hiddenSize1):
            for j in range(hiddenSize2):
                weight2[i][j] = float(data_list[i + inputSize][j])
        for i in range(hiddenSize2):
            for j in range(outputSize):
                weight3[i][j] = float(data_list[i + inputSize + hiddenSize1][j])
        temp = np.array([weight1, weight2, weight3])
        np.save('combined.npy', temp)
    else:
        print("No correct weight file found!")
        return False

    if(dir[len(dir)-4:] ==".csv"):
        arrTemp = np.zeros((40, 40))
        with open(dir, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            if(len(data_list) == 483):
                #formatted csv: list how many cols and rows there are
                highest = np.array([0, 0, 0])
                arr = np.zeros((int(data_list[1][1]),int(data_list[0][1])))
                for i in range(0, int(data_list[1][1])):
                    for j in range(0, int(data_list[0][1])):
                        arr[i][j] = float(data_list[i + 3][j])
                        if arr[i][j] > float(highest[0]):
                            highest[0] = int(data_list[i + 3][j])
                            highest[1] = i
                            highest[2] = j
                indexY = int(highest[1] - 40 / 2)
                indexX = int(highest[2] - 40 / 2)

                for i in range(40):
                    for j in range(40):
                        arrTemp[i][j] = float(data_list[i + 3 + indexY][j+indexX])

            elif(len(data_list) == 480):
                # not formatted csv: does not list how many cols and rows there are
                arr = np.zeros((480,752))
                highest = np.array([0, 0, 0])
                for i in range(0, 480):
                    for j in range(0, 752):
                        arr[i][j] = float(data_list[i][j])
                        if arr[i][j] > float(highest[0]):
                            highest[0] = int(data_list[i][j])
                            highest[1] = i
                            highest[2] = j
                indexY = int(highest[1] - 40 / 2)
                indexX = int(highest[2] - 40 / 2)
                for i in range(40):
                    for j in range(40):
                        arrTemp[i][j] = float(data_list[i + indexY][j + indexX])

            elif(len(data_list) == 40):
                for i in range(40):
                    for j in range(40):
                        arrTemp[i][j] = float(data_list[i][j])
            else:
                #wrong csv format: has to be either: formatted, raw, or already sampled
                print("Wrong csv format")
                return False

        iArr = filtering(arrTemp)
        layer1 = sigmoid(np.dot(iArr, weight1))
        layer2 = sigmoid(np.dot(layer1, weight2))
        guess = sigmoid(np.dot(layer2, weight3))

    elif(dir[len(dir)-4:] ==".npy"):
        iArrTemp = np.load(dir)
        if(len(iArrTemp)==40):
            iArr = filtering(iArrTemp)
            layer1 = sigmoid(np.dot(iArr, weight1))
            layer2 = sigmoid(np.dot(layer1, weight2))
            guess = sigmoid(np.dot(layer2, weight3))
        else:
            arrTemp = np.zeros((40, 40))
            highest = np.array([0, 0, 0])
            for i in range(0, 480):
                for j in range(0, 752):
                    if iArrTemp[i][j] > float(highest[0]):
                        highest[0] = iArrTemp[i][j]
                        highest[1] = i
                        highest[2] = j
            indexY = int(highest[1] - 40 / 2)
            indexX = int(highest[2] - 40 / 2)
            for i in range(40):
                for j in range(40):
                    arrTemp[i][j] = float(data_list[i + indexY][j + indexX])
            iArr = filtering(arrTemp)
            layer1 = sigmoid(np.dot(iArr, weight1))
            layer2 = sigmoid(np.dot(layer1, weight2))
            guess = sigmoid(np.dot(layer2, weight3))
    else:
        print("Not a valid file type: Must be .npy or .csv")
        return False

    if (guess >= .5 and state==True):
        print("bad")
    elif(guess < .5 and state==True):
        print("good")

    return guess

def main():
    print("Enter your path to your temperature map: ")
    print("(accepts: row&col labeled .csv, non  row&col labeled .csv, sampled 4040 .csv, full tempature map .npy,and sampled 4040 .npy data)")
    path = input("directory: ")
    print(test(path, False))

"""
example dir:
bad formatted: /Users/joshchung/Desktop/converted/t306p7_x0_y29p04_z19p89_layer40.csv
good formatted: /Users/joshchung/Desktop/converted/t286p4_x0_y50p33_z18p36_layer37.csv

bad sampled 4040 .csv: /Users/joshchung/Desktop/4040testcases/badSampled_t121p0_x0_y48p40_z7p65_layer16.csv
good sampled 4040 .csv: /Users/joshchung/Desktop/4040testcases/Sampled_t411p3_x0_y25p17_z27p03_layer54.csv

bad sampled 4040 .npy: /Users/joshchung/Desktop/np4040/npbadSampled_t149p1_x0_y0p0000_z9p69_layer20.npy
good sampled 4040 .npy: /Users/joshchung/Desktop/np4040/npSampled_t276p8_x0_y25p17_z17p85_layer36.npy
"""
if __name__ == "__main__":
    main()