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
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2
        self.lr = lr
        self.count = 0
        self.weightFile = weightFile

    #input -(w1)> layer1 -(w2)> layer2 -(w3)> output
    def train(self, input, output):
        layer1 = sigmoid(np.dot(input, self.weights1))
        layer2 = sigmoid(np.dot(layer1, self.weights2))
        guess = sigmoid(np.dot(layer2, self.weights3))

        cost = output - guess
        #a = np.dot(layer1.T, (2 * cost * sigmoid_derivative(self.guess)))
        #np.dot(input.T, (np.dot(a, self.weights2.T) * sigmoid_derivative(layer1)))

        #np.dot(layer1.T, (2 * (output - self.guess) * sigmoid_derivative(self.guess)))
        #np.dot(input.T, (np.dot(2 * (output - self.guess) * sigmoid_derivative(self.guess), self.weights2.T) * sigmoid_derivative(layer1)))


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

def main():
    print("here")
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # def __init__(self, inputSize, outputSize, hiddenSize1, hiddenSize2, lr, state, weightFile):
    nn = NeuralNetwork(2,1,4,4,.1, False, 'weights30000.csv')

    for i in range(30000):
        state = random.randint(0, 4)
        if (state == 0):
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

        nn.train(trainingSet, out)

    for i in range(4):
        print(nn.test(X[i]))


if __name__ == "__main__":
    main()