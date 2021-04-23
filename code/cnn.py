# feature extraction
import numpy as np
from numpy import random
from PIL import Image
import csv
import sys

from os import path
import glob

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

#input->w1->hidden1->w2->hidden2->w3->output

class NeuralNetwork:
    def __init__(self, inputSize, outputSize, hiddenSize1, hiddenSize2, lr, state, weightFile):
        self.weights1 = np.random.rand(inputSize, hiddenSize1)
        self.weights2 = np.random.rand(hiddenSize1, hiddenSize2)
        self.weights3 = np.random.rand(hiddenSize2, outputSize)
        self.bias1 = np.random.rand(1,1)
        self.bias2 = np.random.rand(1, 1)
        self.bias3 = np.random.rand(1, 1)
        self.randomize(self.weights1)
        self.randomize(self.weights2)
        self.randomize(self.weights3)
        self.randomize(self.bias1)
        self.randomize(self.bias2)
        self.randomize(self.bias3)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2
        self.lr = lr
        self.count = 0
        self.weightFile = weightFile

        if (state == True):
            self.save()
        else:
            self.load(self.weightFile)

    def save(self):
        prevCount = int(self.weightFile[self.weightFile.find("weights")+7: self.weightFile.find(".csv")])
        name = self.weightFile[:self.weightFile.find("weights")+7] + str(self.count + prevCount) + '.csv'

        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(self.weights1.shape[0]):
                writer.writerow(self.weights1[i])
            for i in range(self.weights2.shape[0]):
                writer.writerow(self.weights2[i])
            for i in range(self.weights3.shape[0]):
                writer.writerow(self.weights3[i])

    def randomize(self, set):
        for i in range(len(set)):
            for j in range(len(set[0])):
                if (random.randint(0, 2) == 0):
                    set[i][j] *= -1

    def load(self, name):
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
        self.weights1 = np.zeros((self.inputSize, self.hiddenSize1))
        self.weights2 = np.zeros((self.hiddenSize1, self.hiddenSize2))
        self.weights3 = np.zeros((self.hiddenSize2, self.outputSize))
        for i in range(self.inputSize):
            for j in range(self.hiddenSize1):
                self.weights1[i][j] = float(data_list[i][j])
        for i in range(self.hiddenSize1):
            for j in range(self.hiddenSize2):
                self.weights2[i][j] = float(data_list[i + self.inputSize][j])
        for i in range(self.hiddenSize2):
            for j in range(self.outputSize):
                self.weights3[i][j] = float(data_list[i + self.inputSize + self.hiddenSize1][j])
    def outline(self, dir):
        if(dir.find(".csv")!=-1):
            with open(dir, newline='') as csvfile:
                data_list = list(csv.reader(csvfile))
                arr = np.array(data_list).astype(np.float)
        else:
            arr = np.load(dir)
        newArr = np.zeros((40,40))
        for i in range(40):
            for j in range(40):
               if(arr[i][j]>=1635 and arr[i][j]<=1645):
                   newArr[i][j] = 1
        return np.array([newArr.flatten()])

    def filtering(self, dir):
        if(dir.find(".csv")!=-1):
            with open(dir, newline='') as csvfile:
                data_list = list(csv.reader(csvfile))
                arr = np.array(data_list).astype(np.float)
        else:
            arr = np.load(dir)
        newArr = np.zeros((len(arr) - 2, len(arr) - 2))
        for i in range(1, len(arr) - 1):
            for j in range(1, len(arr) - 1):
                newArr[i - 1][j - 1] = 5 * arr[i][j] - arr[i - 1][j] - arr[i + 1][j] - arr[i][j - 1] - arr[i][j + 1]
        div = np.amax(newArr) - np.amin(newArr)
        newArr -= np.amin(newArr)
        newArr /= div
        return np.array([newArr.flatten()])

    def getMax(self, a,b,c,d):
        if(b>a):
            a = b
        if(d>c):
            c = d
        if(c>a):
            return c
        return a

    def filteringPooled(self, dir):
        if (dir.find(".csv") != -1):
            with open(dir, newline='') as csvfile:
                data_list = list(csv.reader(csvfile))
                arr = np.array(data_list).astype(np.float)
        else:
            arr = np.load(dir)
        newArr = np.zeros((len(arr) - 2, len(arr) - 2))
        for i in range(1, len(arr) - 1):
            for j in range(1, len(arr) - 1):
                newArr[i - 1][j - 1] = 5 * arr[i][j] - arr[i - 1][j] - arr[i + 1][j] - arr[i][j - 1] - arr[i][j + 1]
        newArrPooled = np.zeros((int(len(newArr)/2),int(len(newArr)/2)))
        for i in range(len(newArrPooled)):
            for j in range(len(newArrPooled)):
                newArrPooled[i][j] = self.getMax(newArr[2*i][2*j],newArr[2*i][2*j+1],newArr[2*i+1][2*j],newArr[2*i+1][2*j+1])
        div = np.amax(newArrPooled) - np.amin(newArrPooled)
        newArrPooled -= np.amin(newArrPooled)
        newArrPooled /= div
        return np.array([newArrPooled.flatten()])

    #input -(w1)> layer1 -(w2)> layer2 -(w3)> output

    def train(self, input, output):
        layer1 = sigmoid(np.dot(input, self.weights1))
        layer2 = sigmoid(np.dot(layer1, self.weights2))
        guess = sigmoid(np.dot(layer2, self.weights3))

        cost = output - guess

        firstChain = 2 * cost * sigmoid_derivative(guess)
        d_weights3 = np.dot(layer2.T, firstChain)

        secondChain = np.dot(firstChain, self.weights3.T) * sigmoid_derivative(layer2)
        d_weights2 = np.dot(layer1.T, secondChain)

        thirdChain = np.dot(secondChain, self.weights2.T) * sigmoid_derivative(layer1)
        d_weights1 = np.dot(input.T, thirdChain)

        self.weights3 += self.lr * d_weights3
        self.weights2 += self.lr * d_weights2
        self.weights1 += self.lr * d_weights1

        self.count += 1

    def test(self, input):
        layer1 = sigmoid(np.dot(input, self.weights1))
        layer2 = sigmoid(np.dot(layer1, self.weights2))
        guess = sigmoid(np.dot(layer2, self.weights3))
        return guess

    def dataOutput(self,name):
        if(name.find("bad")!=-1):
            return np.array([1])
        else:
            return np.array([0])
    def initCross(self):
        path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/CrossValidationData.csv"
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            prevCount = int(self.weightFile[self.weightFile.find("weights") + 7: self.weightFile.find(".csv")])
            for groupSkip in range(1, 11):
                check = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupSkip) + "/*.csv")
                badcount = 0
                badcountcorrect = 0
                goodcount = 0
                goodcountcorrect = 0
                for item in check:
                    if (item.find("bad") != -1):
                        if (self.test(self.filteringPooled(item)) > .5):
                            badcountcorrect += 1
                        badcount += 1
                    else:
                        if (self.test(self.filteringPooled(item)) < .5):
                            goodcountcorrect += 1
                        goodcount += 1
                thisRow = [str(prevCount + self.count), "Test group:" + str(groupSkip), "Total correct:",
                           str((badcountcorrect + goodcountcorrect) / (badcount + goodcount)), "Bad correct:",
                           str(badcountcorrect) + "/" + str(badcount), "Good correct:",
                           str(goodcountcorrect) + "/" + str(goodcount)]
                writer.writerow(thisRow)

    def crossVal(self):
        path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/CrossValidationData.csv"
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            prevCount = int(self.weightFile[self.weightFile.find("weights") + 7: self.weightFile.find(".csv")])
            for groupSkip in range(1, 11):
                for groupTrain in range(1, 11):
                    if (groupSkip == groupTrain):
                        continue
                    fname = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupTrain) + "/*.csv")
                    while (len(fname) != 0):
                        index = random.randint(0,len(fname))
                        dir = fname[index]
                        self.train(self.filteringPooled(dir),self.dataOutput(dir))
                        fname.pop(index)
                check = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupSkip) + "/*.csv")
                badcount = 0
                badcountcorrect = 0
                goodcount = 0
                goodcountcorrect = 0
                for item in check:
                    if (item.find("bad")!=-1):
                        if (self.test(self.filteringPooled(item)) > .5):
                            badcountcorrect += 1
                        badcount += 1
                    else:
                        if (self.test(self.filteringPooled(item)) < .5):
                            goodcountcorrect += 1
                        goodcount += 1
                """
                col = "Test group: " + str(groupSkip)
                col1 = "Total correct:" + str(badcountcorrect + goodcountcorrect) + "/" + str(badcount + goodcount)
                col2 = "Bad correct: " + str(badcountcorrect) + "/" + str(badcount)
                col3 = "Good correct: " + str(goodcountcorrect) + "/" + str(goodcount)
                """
                thisRow = [str(prevCount + self.count),"Test group:" + str(groupSkip), "Total correct:", str((badcountcorrect + goodcountcorrect)/(badcount + goodcount)), "Bad correct:", str(badcountcorrect) + "/" + str(badcount), "Good correct:", str(goodcountcorrect) + "/" + str(goodcount)]
                writer.writerow(thisRow)

    def testCases4040(self,which):
        fname = glob.glob("/Users/joshchung/Desktop/testCasesFinal/*.npy")
        g = 0
        b = 0
        gc = 0
        bc = 0
        for item in fname:
            if(which == "filter" or which == "dsfilter"):
                guess = self.test(self.filtering(item))
            if(which == "pooled" or which == "dspooled"):
                guess = self.test(self.filteringPooled(item))
            if(which == "normal"):
                guess = self.test(self.standard(np.load(item)))
            if (item.find("bad") != -1):
                b += 1
                if (guess > .50):
                    bc += 1
                elif(which.find("ds")==-1):
                    print(item[32:], guess)
            else:
                g += 1
                if (guess < .50):
                    gc += 1
                elif(which.find("ds")==-1):
                    print(item[32:], guess)

        print("Bad" + str(bc) + "/" + str(b))
        print("good" + str(gc) + "/" + str(g))

    def standard(self,arr):
        high = np.amax(arr)
        low = np.amin(arr)
        arr -= low
        arr /= (high - low)
        #print(np.array([arr.flatten()]))
        return np.array([arr.flatten()])

    def testCases4040Cross(self):
        fname = glob.glob("/Users/joshchung/Desktop/4040testcases/*.csv")
        g = 0
        b = 0
        gc = 0
        bc = 0
        for item in fname:
            guess = self.test(self.filtering(item))

            if (item.find("bad") != -1):
                b += 1
                if (guess > .50):
                    bc += 1
                else:
                    print(item[39:], guess)
            else:
                g += 1
                if (guess < .50):
                    gc += 1
                else:
                    print(item[39:], guess)

        print("Bad" + str(bc) + "/" + str(b))
        print("good" + str(gc) + "/" + str(g))

def main():
    print("here")
    path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/"
    nnPooled = NeuralNetwork(361, 1, 75, 25, .05, False, path + 'filteringpooledFinalweights0.csv')
    nnPooled.initCross()
    nnFiltered = NeuralNetwork(1444, 1, 75, 15, .01, False, path + 'filteringFinalweights760500.csv')

    nnNormal = NeuralNetwork(1600, 1, 75, 16, .01, False, path + '4040randomweights177000.csv')

    print(a)
    for i in range(0):
        print(i)
        if(i == 30):
            nnNormal.lr = .01
        if(i==100):
            nnNormal.lr = .001
        nnNormal.crossVal()

    print("\nfilter only")
    nnFiltered.testCases4040("filter")



"""def filtering(arr):
    newArr = np.zeros((len(arr)-2,len(arr)-2))
    for i in range(1,len(arr)-1):
        for j in range(1, len(arr)-1):
            newArr[i-1][j-1] = 5*arr[i][j]-arr[i-1][j]-arr[i+1][j]-arr[i][j-1]-arr[i][j+1]
    div = np.amax(newArr)-np.amin(newArr)
    newArr -= np.amin(newArr)
    newArr /= div
    return newArr"""


if __name__ == "__main__":
    main()