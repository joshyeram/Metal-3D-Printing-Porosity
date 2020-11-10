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
    def __init__(self, inputSize, outputSize, hiddenSize, lr, state, weightFile):
        self.weights1 = np.random.rand(inputSize, hiddenSize)
        self.weights2 = np.random.rand(hiddenSize, outputSize)
        self.bias = np.array(np.random.rand(2))
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
        self.count = 0
        self.weightFile = weightFile

        if (state == True):
            self.save()
        else:
            self.load(self.weightFile)

    def train(self, input, output, dir):
        layer1 = sigmoid(np.dot(input, self.weights1)+self.bias[0])
        self.guess = sigmoid(np.dot(layer1, self.weights2)+self.bias[1])

        d_weights2 = self.lr * np.dot(layer1.T, (2 * (output - self.guess) * sigmoid_derivative(self.guess)))
        db2 = np.sum(sigmoid_derivative(self.guess) * (output - self.guess),axis = 0, keepdims=True)
        d_weights1 = self.lr * np.dot(input.T, (np.dot(2 * (output - self.guess) * sigmoid_derivative(self.guess), self.weights2.T) * sigmoid_derivative(layer1)))
        grad_1 = np.sum(d_weights2.T, axis=0, keepdims=True)
        db1 = np.sum(sigmoid_derivative(layer1) * grad_1,axis = 1, keepdims=True)

        """ 
        weighted changes/back propogation
        
        if(dir.find("bad")==-1):
            if (self.guess > .2):
                d_weights1 *= .2
                d_weights2 *= .2
            else:
                d_weights1 *= .1
                d_weights2 *= .1
        else:
            if(self.guess < .2):
                d_weights1 *= .8
                d_weights2 *= .8
        """

        """
        focal loss
        
        error = output - self.guess
        #1-guess
        if(dir.find("bad")==-1):
            d_weights1 *= -math.log(error, 10) * .2
            d_weights2 *= -math.log(error, 10) * .2
        else:
            d_weights1 *= -math.log(self.guess, 10) * 2
            d_weights2 *= -math.log(self.guess, 10) * 2
        """

        """
        smote: Synthetic Minority Oversampling Technique
        """

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.bias[0] += self.lr *.1 * db1
        self.bias[1] += self.lr *.1 * db2
        self.count += 1

    def test(self, input):
        layer1 = sigmoid(np.dot(input, self.weights1)+self.bias[0])
        self.guess = sigmoid(np.dot(layer1, self.weights2)+self.bias[1])

        return self.guess

    def save(self):
        prevCount = int(self.weightFile[self.weightFile.find("weights")+7: self.weightFile.find(".csv")])
        with open(self.weightFile[:self.weightFile.find("weights")+7] + str(self.count + prevCount) + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(self.weights1.shape[0]):
                writer.writerow(self.weights1[i])
            for i in range(self.weights2.shape[0]):
                writer.writerow(self.weights2[i])
            for i in range(2):
                writer.writerow([self.bias[i]])

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
        for i in range(2):
            self.bias[i] = float(data_list[i + self.inputSize+ self.hiddenSize][0])
        self.weights1 = np.array(w1)
        self.weights2 = np.array(w2)

    def dataInput(self, name):
        data = np.zeros(self.inputSize, dtype=np.float128)
        count = 0
        high = 0
        low = 1500
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            for i in range(30):
                for j in range(30):
                    data[count] = float(data_list[i][j])
                    if(data[count]>high):
                        high = data[count]
                    elif(data[count]<low):
                        low = data[count]
                    #data[count] = (data[count]-1129)/(2107-1129)
                    count += 1

        for i in range(len(data)):
            #data[i] =  (data[i] - 1129) / (2107-1129)
            data[i] = (data[i] - low) / (high - low)
        set = np.array([data])

        #return np.array([data])
        return set
    def dataInputCompact(self, name):
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            arr = np.array(data_list).astype(np.float).flatten()
            high = np.amax(arr)
            low = np.amin(arr)
            arr -= low
            arr /= (high-low)
        return arr

    def getOut(self,name):
        if(name.find("bad")!=-1):
            return np.array([1])
        else:
            return np.array([0])

    def crossVal(self):
        with open('CrossVal.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            prevCount = int(self.weightFile[self.weightFile.find("weights")+7: self.weightFile.find(".csv")])
            writer.writerow([str(prevCount + self.count)])
            for groupSkip in range(1, 11):
                for groupTrain in range(1, 11):
                    if (groupSkip == groupTrain):
                        continue
                    fname = glob.glob("/Users/joshchung/Desktop/crossValcopy/group" + str(groupTrain) + "/*.csv")
                    while (len(fname) != 0):
                        index = random.randint(0,len(fname))
                        dir = fname[index]
                        self.train(self.dataInput(dir),self.getOut(dir),dir)
                        fname.pop(index)
                check = glob.glob("/Users/joshchung/Desktop/crossValcopy/group" + str(groupSkip) + "/*.csv")
                badcount = 0
                badcountcorrect = 0
                goodcount = 0
                goodcountcorrect = 0
                for item in check:
                    if (item.find("bad")!=-1):
                        if (self.test(self.dataInput(item)) > .7):
                            badcountcorrect += 1
                        badcount += 1
                    else:
                        if (self.test(self.dataInput(item)) < .2):
                            goodcountcorrect += 1
                        goodcount += 1
                col = "Test group: " + str(groupSkip)
                col1 = "Total correct: " + str(badcountcorrect + goodcountcorrect) + "/" + str(badcount + goodcount)
                col2 = "Bad correct: " + str(badcountcorrect) + "/" + str(badcount)
                col3 = "Good correct: " + str(goodcountcorrect) + "/" + str(goodcount)
                writer.writerow([col, col1, col2, col3])

def main():
    lr = .1

    #nn = NeuralNetwork(900, 1, 50, lr, False, '0to1scaledweights3982506.csv')
    nn = NeuralNetwork(900, 1, 200, lr, False, 'osLargersHiddenweights537120.csv')
    #nn.save()

    """
    for i in range(20):
        print(i)
        #nnTesting.crossVal()
        nn.crossVal()

    nn.save()
    """

    fname = glob.glob("/Users/joshchung/Desktop/testCases/*.csv")
    g = 0
    b = 0
    gc = 0
    bc = 0
    for item in fname:
        guess = nn.test(nn.dataInput(item))
        print(item[35:], guess)
        if(item.find("bad")!=-1):
            b+=1
            if(guess>.45):
                bc+=1
        else:
            g+=1
            if (guess < .45):
                gc += 1

    print("Bad"+ str(bc)+"/"+str(b))
    print("good" + str(gc) + "/" + str(g))


if __name__ == "__main__":
    main()
