import numpy as np
from numpy import random
import csv
from os import path
import glob


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x) + .0001


class NeuralNetwork:
    """
    50 test cases
        25 bad
        25 good
    1509 train data
    10 fold cross validation
    Around 150 cases per batch


    worst case: sumation of all 6400 elements with their weights are = 1

    """

    def __init__(self, inputSize, outputSize, hiddenSize, lr, state, weightFile):
        self.weights1 = np.random.rand(inputSize, hiddenSize)
        self.weights2 = np.random.rand(hiddenSize, outputSize)
        for i in range(inputSize):
            for j in range(hiddenSize):
                if(random.randint(0,2)==0):
                    self.weights1[i][j]*=-1
        for i in range(hiddenSize):
            for j in range(outputSize):
                if(random.randint(0,2)==0):
                    self.weights2[i][j]*=-1
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.guess = np.zeros(outputSize)
        self.lr = lr
        self.count = 1
        self.weightFile = weightFile

        if (state == True):
            self.save()
        else:
            self.load(self.weightFile)

        with open('porosityName.csv', newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
        self.bad = data_list
    def trainTest(self,input, output):
        layer1 = sigmoid(np.dot(input, self.weights1))
        self.guess = sigmoid(np.dot(layer1, self.weights2))

        d_weights2 = self.lr * np.dot(layer1.T, (2 * (output - self.guess) * sigmoid_derivative(self.guess)))
        d_weights1 = self.lr * np.dot(input.T, (np.dot(2 * (output - self.guess) * sigmoid_derivative(self.guess), self.weights2.T) * sigmoid_derivative(layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.count += 1

    def train(self, input, output):
        # print(self,input.shape)
        # print(self.weights1.shape)
        layer1 = sigmoid(np.dot(input, self.weights1))
        self.guess = sigmoid(np.dot(layer1, self.weights2))

        # print("forward", layer1.shape,layer1)
        # print("guess",self.guess)
        # print(layer1)
        # print("dw1 elemetnts",sigmoid_derivative(self.guess),(2*(output-self.guess) * sigmoid_derivative(self.guess)).shape)
        d_weights2 = self.lr * np.dot(layer1.T, (2 * (output - self.guess) * sigmoid_derivative(self.guess)))
        d_weights1 = self.lr * np.dot(input.T, (np.dot(2 * (output - self.guess) * sigmoid_derivative(self.guess),
                                                       self.weights2.T) * sigmoid_derivative(layer1)))
        # print("backward", d_weights2,d_weights1)

        #print(self.guess, output, output-self.guess)
        # print(d_weights2[0][0])
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.count += 1
        #print(self.guess, output, output-self.guess,d_weights2.T,)
        # print("backward", d_weights2, d_weights1)
        # print(" ")

    def trainCrossVal(self):
        with open('CrossVal.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            prevCount = int(self.weightFile[7: self.weightFile.find(".csv")])
            writer.writerow([str(prevCount + self.count)])
            groupPath = "/Users/joshchung/Desktop/crossVal/group"
            for groupSkip in range(1, 11):
                for groupTrain in range(1, 11):
                    if (groupSkip == groupTrain):
                        continue
                    fname = glob.glob(groupPath + str(groupTrain) + "/*.csv")
                    while (len(fname) != 0):
                        index = random.randint(0, len(fname))
                        point = fname[index].find("p" + str(groupTrain) + "/") + 3
                        if (groupTrain == 10):
                            point += 1
                        name = fname[index][point:]
                        # print(fname[index])
                        # print(self.dataInput(fname[index],groupTrain),self.getOutput(name))
                        self.train(self.dataInput(fname[index], groupTrain), self.getOutput(name))
                        fname.pop(index)
                    # print(groupSkip,groupTrain)
                check = glob.glob(groupPath + str(groupSkip) + "/*.csv")
                badcount = 0
                badcountcorrect = 0
                goodcount = 0
                goodcountcorrect = 0
                for item in check:
                    if (item[item.find("Sampled") - 1:item.find("Sampled")] == "1"):
                        if (self.test(self.dataInput(item, groupSkip)) > .9):
                            badcountcorrect += 1
                        badcount += 1
                    else:
                        if (self.test(self.dataInput(item, groupSkip)) < .1):
                            goodcountcorrect += 1
                        goodcount += 1
                    # print(item[item.find("p"+str(groupSkip)+"/") + 3:])
                col = "Test group: " + str(groupSkip)
                col1 = "Total correct: " + str(badcountcorrect + goodcountcorrect) + "/" + str(badcount + goodcount)
                col2 = "Bad correct: " + str(badcountcorrect) + "/" + str(badcount)
                col3 = "Good correct: " + str(goodcountcorrect) + "/" + str(goodcount)
                writer.writerow([col, col1, col2, col3])

    def test(self, input):
        layer1 = sigmoid(np.dot(input, self.weights1))
        self.guess = sigmoid(np.dot(layer1, self.weights2))
        return self.guess

    def save(self):
        prevCount = int(self.weightFile[7: self.weightFile.find(".csv")])
        with open('weights' + str(self.count + prevCount) + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(self.weights1.shape[0]):
                writer.writerow(self.weights1[i])
            for i in range(self.weights2.shape[0]):
                writer.writerow(self.weights2[i])

    def load(self, name):
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
        w1 = [[0.] * self.hiddenSize for i in range(self.inputSize)]
        w2 = [[0.] * self.outputSize for i in range(self.hiddenSize)]
        for i in range(self.inputSize):
            for j in range(self.hiddenSize):
                w1[i][j] = float(data_list[i][j])
        for i in range(self.hiddenSize):
            for j in range(self.outputSize):
                w2[i][j] = float(data_list[i + self.inputSize][j])
        self.weights1 = np.array(w1)
        self.weights2 = np.array(w2)

    def dataRot(self, name, count):
        data = np.zeros((80, 80))
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            for i in range(80):
                for j in range(80):
                    data[i][j] = float(data_list[i][j]) * .0001
        ct = 0
        temp = np.rot90(data, count)
        dataRe = np.zeros(self.inputSize)
        for i in range(80):
            for j in range(80):
                dataRe[ct] = temp[i][j]
                ct += 1
        return np.array([dataRe])

    def testFinal(self, name):
        print(name)
        avg = 0.
        good = 0
        bad = 0
        for i in range(4):
            x = self.test(self.dataRot(name, i))
            print(x)
            if (x < .01):
                good += 3
            elif (x > .7):
                bad += 1
            else:
                good += 1
            avg += 1
        print(bad, good)
        if (good > bad):
            print("prob good")
            return "prob good"
        elif (bad > good):
            print("prob bad")
            return "prob bad"
            # print(name,self.test(self.dataRot(name,i)))
        print("dont know", avg / 4)

    def dataInput(self, name, group):
        data = np.zeros(self.inputSize, dtype=np.float128)
        count = 0
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            for i in range(30):
                for j in range(30):
                    data[count] = float(data_list[i][j])
                    data[count] = (data[count]-1129)/(2107-1129)
                    count += 1
        set = np.array([data])
        norm = set/np.sqrt(4470647879204)
        #print(set,normal_array) 4470647879204
        #print(norm)
        return set

    def getOutput(self, name):
        if (name[:1] == "1"):
            return np.array([1])  # there is porosity
        else:
            return np.array([0])  # there is no porosity
    def getOut(self,name):
        if(name.find("bad")!=-1):
            return np.array([1])
        else:
            return np.array([0])

def main():
    input = 6400
    hidden = 300
    output = 1
    lr = .2

    #nn = NeuralNetwork(input, output, hidden, lr, False, 'weights0.csv')
    nnTesting = NeuralNetwork(900, 1, 100, lr, True, 'weights0.csv')
    #print(nnTesting.test(nnTesting.dataInput("/Users/joshchung/Desktop/temp/90bad36.csv", 1)))
    nnTesting.save()
    for i in range(5):
        print(i)
        #print(nnTesting.test(nnTesting.dataInput("/Users/joshchung/Desktop/temp/90bad36.csv", 1)))
        pathi = glob.glob("/Users/joshchung/Desktop/temp/*.csv")
        while(len(pathi)!=0):
            ro = random.randint(0,len(pathi))
            fname = pathi[ro]
            pathi.pop(ro)
            #print(fname)
            nnTesting.train(nnTesting.dataInput(fname,1),nnTesting.getOut(fname))

    #nnTesting.save()
    print(nnTesting.test(nnTesting.dataInput("/Users/joshchung/Desktop/temp/90bad36.csv", 1)))
    print()
    for i in range(1,1400):
        print(nnTesting.test(nnTesting.dataInput("/Users/joshchung/Desktop/temp/good"+str(i)+".csv", 1)))
    for i in range(1,71):
        print(nnTesting.test(nnTesting.dataInput("/Users/joshchung/Desktop/temp/bad"+str(i)+".csv", 1)))


if __name__ == "__main__":
    main()

"""
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(2,1,4,.1,False,'weights30000.csv')
    for i in range(30000):
        state = random.randint(0,4)
        if(state==0):
            trainingSet = np.array([X[0]])
            out = np.array([y[0]])
        elif (state == 1):
            trainingSet = np.array([X[1]])
            out = np.array([y[1]])
        elif (state == 2):
            trainingSet = np.array([X[2]])
            out = np.array([y[2]])
        else:
            trainingSet = np.array([X[3]])
            out = np.array([y[3]])
        nn.train(trainingSet,out)
    for i in range(4):
        print(nn.test(X[i]))
    nn.save()
"""
