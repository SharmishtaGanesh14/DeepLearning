import numpy as np
from fontTools.cffLib import writeCard16


def sigmoid(self, z):
    sig_z = 1 / (1 + np.exp(-z))
    return np.array(sig_z)


def tanh(self, z):
    tanh_z = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return np.array(tanh_z)


def relu(self, z):
    ReLU = (z + np.abs(z)) / 2
    return np.array(ReLU)


def leaky_relu(self, z):
    leaky_ReLU = np.maximum(0.01 * z, z)
    return np.array(leaky_ReLU)


def softmax(self, z):
    exp_z = np.exp(z)
    softmax = exp_z / np.sum(np.exp(z), axis=0, keepdims=True)
    return np.array(softmax)


def node(input1,input2,operation):
    if operation == 'add':
        return input1 + input2
    if operation == 'max':
        return max(input1, input2)
    if operation == 'multiply':
        return input1 * input2


def main():
    x1,w1,x2,w2=2,3,4,5
    f1=x1*w1
    f2=x2*w2
    f3=f1+f2

if __name__ == "__main__":
    main()