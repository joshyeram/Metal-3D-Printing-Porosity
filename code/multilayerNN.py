import numpy as np
from numpy import random
import csv
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

    def trainMinibatch(self,inputDir):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        tempDw1 = 0
        tempDw2 = 0
        tempDw3 = 0
        counter = 0
        while (len(inputDir) != 0):
            for i in range(50):
                if(len(inputDir) == 0):
                    break
                index = random.randint(0, len(inputDir))
                dir = inputDir[index]
                inputDir.pop(index)

                layer1 = sigmoid(np.dot(self.dataInputCompact(dir), self.weights1))
                layer2 = sigmoid(np.dot(layer1, self.weights2))
                guess = sigmoid(np.dot(layer2, self.weights3))
                output = self.dataOutput(dir)
                cost = output - guess

                firstChain = 2 * cost * sigmoid_derivative(guess)
                tempDw3 += np.dot(layer2.T, firstChain)

                secondChain = np.dot(firstChain, self.weights3.T) * sigmoid_derivative(layer2)
                tempDw2 += np.dot(layer1.T, secondChain)

                thirdChain = np.dot(secondChain, self.weights2.T) * sigmoid_derivative(layer1)
                tempDw1 += np.dot(self.dataInputCompact(dir).T, thirdChain)

                counter+=1
                self.count += 1

                if (guess >= .5 and output == 1):
                    tp += 1
                elif (guess >= .5 and output == 0):
                    fp += 1
                elif (guess < .5 and output == 1):
                    fn += 1
                elif (guess < .5 and output == 0):
                    tn += 1

            currFMoffset = 1 - self.fmeasure(tp, fp, fn, tn)
            #print(currFMoffset)


            self.weights3 += self.lr * currFMoffset * tempDw3/counter
            self.weights2 += self.lr * currFMoffset * tempDw2/counter
            self.weights1 += self.lr * currFMoffset * tempDw1/counter
            counter = 0
            tempDw3 = 0
            tempDw2 = 0
            tempDw1 = 0
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            #print(currFM)

    def trainWithFmeasure(self, inputDir):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        collected = 0
        currArray = []

        while(len(inputDir)!=0):
            index = random.randint(0, len(inputDir))
            dir = inputDir[index]
            currArray.append(dir)
            inputDir.pop(index)

            layer1 = sigmoid(np.dot(self.dataInputCompact(dir), self.weights1))
            layer2 = sigmoid(np.dot(layer1, self.weights2))
            guess = sigmoid(np.dot(layer2, self.weights3))
            output = self.dataOutput(dir)

            if (guess >= .2 and output == 1):
                tp += 1
            if (guess >= .2 and output == 0):
                fp += 1
            if (guess < .2 and output == 1):
                fn += 1
            if (guess < .2 and output == 0):
                tn += 1
            collected +=1

            if(collected>=50 or len(inputDir)==0):
                currFM = self.fmeasure(tp, fp, fn, tn)

                tempD1 = 0
                tempD2 = 0
                tempD3 = 0
                while (len(currArray) != 0):
                    temp = currArray.pop(0)
                    layer1 = sigmoid(np.dot(self.dataInputCompact(temp), self.weights1))
                    layer2 = sigmoid(np.dot(layer1, self.weights2))
                    guess = sigmoid(np.dot(layer2, self.weights3))

                    firstChain = 2 * (1 - currFM) * sigmoid_derivative(guess)
                    tempD3 += np.dot(layer2.T, firstChain)

                    secondChain = np.dot(firstChain, self.weights3.T) * sigmoid_derivative(layer2)
                    tempD2 += np.dot(layer1.T, secondChain)

                    thirdChain = np.dot(secondChain, self.weights2.T) * sigmoid_derivative(layer1)
                    tempD1 += np.dot(self.dataInputCompact(temp).T, thirdChain)

                self.weights3 += self.lr * (1.0 / collected) * tempD3
                self.weights2 += self.lr * (1.0 / collected) * tempD2
                self.weights1 += self.lr * (1.0 / collected) * tempD1
                collected = 0

            self.count += 1

    def fmeasure(self,tp, fp, fn, tn):
        if(tp+fp == 0):
            return 0
        f1 = (tp*1.0)/(tp + .5 * (fp+fn))
        b = .5
        f2 = ((1+b*b) * tp) / (((1+b*b) * tp)+(b*b*fn)+fp)
        """print("tp:"+str(tp)+"  fp:"+str(fp)+"  fn:"+str(fn)+"  tn:"+str(tn))
        print("f1:"+ str(f1))
        print("f pt5:" + str(f2))"""
        return f2

    def crossValFM(self):
        path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/CrossValidationData.csv"
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            prevCount = int(self.weightFile[self.weightFile.find("weights") + 7: self.weightFile.find(".csv")])
            for groupSkip in range(1, 11):
                for groupTrain in range(1, 11):
                    if (groupSkip == groupTrain):
                        continue
                    fname = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupTrain) + "/*.csv")
                    self.trainMinibatch(fname)
                check = glob.glob("/Users/joshchung/Desktop/cross/g" + str(groupSkip) + "/*.csv")
                badcount = 0
                badcountcorrect = 0
                goodcount = 0
                goodcountcorrect = 0
                for item in check:
                    if (item.find("bad") != -1):
                        if (self.test(self.dataInputCompact(item)) >= .5):
                            badcountcorrect += 1
                        badcount += 1
                    else:
                        if (self.test(self.dataInputCompact(item)) < .5):
                            goodcountcorrect += 1
                        goodcount += 1
                thisRow = [str(prevCount + self.count), "Test group:" + str(groupSkip), "Total correct:",
                           str((badcountcorrect + goodcountcorrect) / (badcount + goodcount)), "Bad correct:",
                           str(badcountcorrect) + "/" + str(badcount), "Good correct:",
                           str(goodcountcorrect) + "/" + str(goodcount)]
                writer.writerow(thisRow)

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
    def standard(self,arr):
        high = np.amax(arr)
        low = np.amin(arr)
        arr -= low
        arr /= (high - low)
        return np.array([arr])


def main():
    print("here")

    # def __init__(self, inputSize, outputSize, hiddenSize1, hiddenSize2, lr, state, weightFile): fm for bad
    path = "/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/data/"

    nnIR = NeuralNetwork(2600,1,250,25,.03, False, path+'IRbareweights65940.csv')
    for i in range(30):
        print(i)
        patho = glob.glob("/Users/joshchung/Desktop/nparrays/*.npy")
        for paths in patho:
            p = np.load(paths).flatten()
            nnIR.train(nnIR.standard(p),nnIR.dataOutput(paths))

    nnIR.save()
    patho = glob.glob("/Users/joshchung/Desktop/nparrays/*.npy")
    right = 0
    wrong = 0
    for paths in patho:
        p = np.load(paths).flatten()
        temp = nnIR.test(nnIR.standard(p))
        if(temp>=.5 and nnIR.dataOutput(paths)<.5):
            print(temp, paths)
            wrong+=1
        elif (temp <= .5 and nnIR.dataOutput(paths) > .5):
            print(temp, paths)
            wrong += 1
        else:
            right+=1
    print("correct: "+ str(right))
    print("incorrect: " + str(wrong))
    print("frac: " + str(right/(right+wrong)))

if __name__ == "__main__":
    main()
