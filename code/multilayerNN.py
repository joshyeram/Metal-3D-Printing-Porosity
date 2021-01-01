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

    def trainWithFmeasure(self, input, output):
        tp = 0
        fp = 0
        fn = 0

        for path in input:
            layer1 = sigmoid(np.dot(self.dataInputCompact(path), self.weights1))
            layer2 = sigmoid(np.dot(layer1, self.weights2))
            guess = sigmoid(np.dot(layer2, self.weights3))
            tp += guess * output.item()
            fp += guess * (1-output.item())
            fn += (1-guess) * output.item()

        fm = ((1 + 2 * 2) * tp) / ((1 + 2 * 2) * tp + (2*2)*fn + fp)
        dfm = 0


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

        self.count += 10

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

    def dataInputCompact40to30(self, name):
        with open(name, newline='') as csvfile:
            data_list = list(csv.reader(csvfile))
            conv = np.array(data_list).astype(np.float)
            arr = np.zeros((30,30))
            for i in range(0,30):
                for j in range(0, 30):
                    arr[i][j] = conv[i+5][j+5]
            high = np.amax(arr)
            low = np.amin(arr)
            arr -= low
            arr /= (high-low)
            temp = arr.flatten()
        return np.array([temp])

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
 #f-measure
    def testCases4040out(self):
        fname = glob.glob("/Users/joshchung/Desktop/4040testCases/*.csv")
        g = 0
        b = 0
        gc = 0
        bc = 0
        for item in fname:
            guess = self.test(self.dataInputOutline(item))

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

    def testCasesFinal(self, n4040, n3030):
        fname = glob.glob("/Users/joshchung/Desktop/4040testCases/*.csv")
        g = 0
        b = 0
        gc = 0
        bc = 0
        for item in fname:
            temp4040 = n4040.test(self.dataInputCompact(item))
            temp3030 = n3030.test(self.dataInputCompact40to30(item))
            z = 0
            if (item.find("p0000_0_layer1") != -1 or item.find("z0_layer1") != -1):
                z = 0
            else:
                temp = item[item.find("z") + 1: item.find("_layer")]
                z = float(temp.replace("p", "."))
            inpu = np.array([temp4040.item(), temp3030.item(), z])
            real = np.array([inpu])
            guess = self.test(real)

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

    # 4040
    # 3030
    # z height



    # 3 4 4 1
    # p0000_0_layer1 = 0
    # z0_layer1 = 0
    # find z later
    # cut off _layer
    # replace p with .
    # float it
    def testing(self):
        for groupSkip in range(1, 11):
            for groupTrain in range(1, 11):
                if (groupSkip == groupTrain):
                    continue
                fname = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupTrain) + "/*.csv")
                while (len(fname) != 0):
                    index = random.randint(0, len(fname))
                    dir = fname[index]
                    if(dir.find("p0000_0_layer1") != -1 or dir.find("z0_layer1")!= -1):
                        print(0)
                    else:
                        temp = dir[dir.find("z")+1: dir.find("_layer")]
                        print(float(temp.replace("p",".")))
                    fname.pop(index)
    def combine(self, n4040, n3030):
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
                        index = random.randint(0, len(fname))
                        dir = fname[index]
                        temp4040 = n4040.test(self.dataInputCompact(dir))
                        temp3030 = n3030.test(self.dataInputCompact40to30(dir))
                        z = 0
                        if (dir.find("p0000_0_layer1") != -1 or dir.find("z0_layer1") != -1):
                            z = 0
                        else:
                            temp = dir[dir.find("z") + 1: dir.find("_layer")]
                            z = float(temp.replace("p", "."))

                        input4040 = 0
                        input3030 = 0
                        if(temp4040.item() > .3):
                            input4040 = 1
                        if (temp3030.item() > .3):
                            input3030 = 1
                        inpu = np.array([input4040,input3030,z])
                        """
                        if(self.dataOutput(dir)==0 and (temp4040.item()>.3 or temp3030.item()>.3 )):
                            print(inpu, dir )
                        elif(self.dataOutput(dir)==1 and (temp4040.item()<.7 or temp3030.item()<.7 )):
                            print(inpu, dir)
                        """
                        real = np.array([inpu])
                        #print(real)
                        self.train(real, self.dataOutput(dir))
                        fname.pop(index)

                check = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupSkip) + "/*.csv")
                badcount = 0
                badcountcorrect = 0
                goodcount = 0
                goodcountcorrect = 0
                for item in check:
                    temp4040 = n4040.test(self.dataInputCompact(item))
                    temp3030 = n3030.test(self.dataInputCompact40to30(item))
                    z = 0
                    if (item.find("p0000_0_layer1") != -1 or item.find("z0_layer1") != -1):
                        z = 0
                    else:
                        temp = item[item.find("z") + 1: item.find("_layer")]
                        z = float(temp.replace("p", "."))
                    input4040 = 0
                    input3030 = 0
                    if (temp4040.item() > .3):
                        input4040 = 1
                    if (temp3030.item() > .3):
                        input3030 = 1
                    inpu = np.array([input4040, input3030, z])
                    real = np.array([inpu])

                    if (item.find("bad") != -1):
                        if (self.test(real) > .7):
                            badcountcorrect += 1
                        badcount += 1
                    else:
                        if (self.test(real) < .2):
                            goodcountcorrect += 1
                        goodcount += 1
                """
                col = "Test group: " + str(groupSkip)
                col1 = "Total correct:" + str(badcountcorrect + goodcountcorrect) + "/" + str(badcount + goodcount)
                col2 = "Bad correct: " + str(badcountcorrect) + "/" + str(badcount)
                col3 = "Good correct: " + str(goodcountcorrect) + "/" + str(goodcount)
                """
                thisRow = [str(prevCount + self.count), "Test group:" + str(groupSkip), "Total correct:",
                           str((badcountcorrect + goodcountcorrect) / (badcount + goodcount)), "Bad correct:",
                           str(badcountcorrect) + "/" + str(badcount), "Good correct:",
                           str(goodcountcorrect) + "/" + str(goodcount)]
                writer.writerow(thisRow)



def main():
    print("here")

    # def __init__(self, inputSize, outputSize, hiddenSize1, hiddenSize2, lr, state, weightFile): fm for bad
    path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/"

    nn4040 = NeuralNetwork(1600,1,75,16,.005, False, path+'4040weights1064700.csv')
    nn3030 = NeuralNetwork(900, 1, 50, 16, .005, False, path + '3030weights2770200.csv')
    nnFinal = NeuralNetwork(3, 1, 4, 4, .1, False, path + 'combinedweights152100.csv')
    nnFinal.testCasesFinal(nn4040,nn3030)
    print()
    nn4040.testCases4040()

    for i in range(0):
        print(i)
        nnFinal.combine(nn4040, nn3030)
        #nn4040.crossVal()
    #nnFinal.save()
    #nn4040.save()


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