import numpy as np
from numpy import random
import csv
from os import path
import glob

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, inputSize, outputSize, hiddenSize1, hiddenSize2, lr, state, weightFile):
        self.weights1 = np.random.rand(inputSize, hiddenSize1)
        self.weights2 = np.random.rand(hiddenSize1, hiddenSize2)
        self.weights3 = np.random.rand(hiddenSize2, outputSize)
        self.randomize(self.weights1)
        self.randomize(self.weights2)
        self.randomize(self.weights3)
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

    def dataInputCompact(self, name):
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            arr = np.array(data_list).astype(np.float).flatten()
            high = np.amax(arr)
            low = np.amin(arr)
            arr -= low
            arr /= (high-low)
        return np.array([arr])

    def dataInputOutline(self, name):
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            arr = np.array(data_list).astype(np.float).flatten()
            for i in range(len(arr)):
                if(arr[i]>=1630 and arr[i]<=1640):
                    arr[i] = 1
                else:
                    arr[i] = 0
        return np.array([arr])

    def dataOutput(self,name):
        if(name.find("bad")!=-1):
            return np.array([1])
        else:
            return np.array([0])

    def crossVal(self):
        path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/CrossValidationData.csv"
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            prevCount = int(self.weightFile[self.weightFile.find("weights") + 7: self.weightFile.find(".csv")])
            #writer.writerow([str(prevCount + self.count)])
            for groupSkip in range(1, 11):
                for groupTrain in range(1, 11):
                    if (groupSkip == groupTrain):
                        continue
                    fname = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupTrain) + "/*.csv")
                    while (len(fname) != 0):
                        index = random.randint(0,len(fname))
                        dir = fname[index]
                        self.train(self.dataInputCompact(dir),self.dataOutput(dir))
                        fname.pop(index)
                check = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupSkip) + "/*.csv")
                badcount = 0
                badcountcorrect = 0
                goodcount = 0
                goodcountcorrect = 0
                for item in check:
                    if (item.find("bad")!=-1):
                        if (self.test(self.dataInputCompact(item)) > .7):
                            badcountcorrect += 1
                        badcount += 1
                    else:
                        if (self.test(self.dataInputCompact(item)) < .2):
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

    def testCases4040(self):
        fname = glob.glob("/Users/joshchung/Desktop/4040testCases/*.csv")
        g = 0
        b = 0
        gc = 0
        bc = 0
        for item in fname:
            guess = self.test(self.dataInputCompact(item))

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
    def testCases3030(self):
        fname = glob.glob("/Users/joshchung/Desktop/testCases/*.csv")
        g = 0
        b = 0
        gc = 0
        bc = 0
        for item in fname:
            guess = self.test(self.dataInputCompact(item))

            if (item.find("bad") != -1):
                b += 1
                if (guess > .15):
                    bc += 1
                else:
                    print(item[35:], guess)
            else:
                g += 1
                if (guess < .15):
                    gc += 1
                else:
                    print(item[35:], guess)

        print("Bad" + str(bc) + "/" + str(b))
        print("good" + str(gc) + "/" + str(g))

def main():
    print("here")

    # def __init__(self, inputSize, outputSize, hiddenSize1, hiddenSize2, lr, state, weightFile):
    path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/"
    nnOutline = NeuralNetwork(1600,1,75,16,.005, False, path+'outlineweights228150.csv')
    nn2 = NeuralNetwork(900,1,50,16,.005, False, path+'weights2770200.csv')
    nn2.testCases3030()
    nnOutline.testCases4040()

    for i in range(0):
        print(i)
        nnOutline.crossVal()
    #nnOutline.save()


if __name__ == "__main__":
    main()






    """
    for i in range(30000):
        x = random.randint(0, 2)
        y = random.randint(0, 2)
        if (x==y):
            nn.train(np.array([[x,y]]), np.array([[1]]))
        else:
            nn.train(np.array([[x, y]]), np.array([[0]]))
    nn.save()
    for i in range(20):
        x = random.randint(0, 2)
        y = random.randint(0, 2)
        print(x,y)
        print(nn.test(np.array([[x, y]])))
        print()
    """