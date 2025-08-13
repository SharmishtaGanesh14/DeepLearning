# Date created: 06/08/2025
# Date updated: 13/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan

# Aim: Implement a 1-layer (input - output layer) neural network from
# scratch for the following dataset. This includes implementing forward
# and backward passes from scratch. Print the training loss and plot
# it over 1000 iterations.

import numpy as np
import matplotlib.pyplot as plt

def init_weights(n_in, n_out, activation):
    if activation.lower() == 'relu':
        std = np.sqrt(2 / n_in)  # He normal init
        return np.random.randn(n_in, n_out) * std
    elif activation.lower() in ['sigmoid', 'tanh', 'softmax']:
        limit = np.sqrt(6 / (n_in + n_out))  # Xavier uniform init
        return np.random.uniform(-limit, limit, (n_in, n_out))
    else:
        raise ValueError(f"Unknown activation: {activation}")
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(float)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)
def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def softmax_deriv(x):
    s = softmax(x)
    return s * (1 - s)

class FeedForwardNet:
    def __init__(self, input_size, hidden_layers, output_size, activations):
        layer_sizes = [input_size] + hidden_layers + [output_size]
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Activations list length must match number of layers")
        self.weights, self.biases = [], []
        self.activations, self.act_derivs = [], []
        self.last_activation = activations[-1].lower()
        self.ACT_FUNCS={
            'relu': (relu, relu_deriv),
            'sigmoid': (sigmoid, sigmoid_deriv),
            'tanh': (tanh, tanh_deriv),
            'softmax': (softmax, softmax_deriv) 
        }
        for i in range(len(layer_sizes) - 1):
            act_name = activations[i].lower()
            if act_name not in self.ACT_FUNCS:
                raise ValueError(f"Unsupported activation: {act_name}")
            act, act_deriv = self.ACT_FUNCS[act_name]
            self.activations.append(act)
            self.act_derivs.append(act_deriv)
            w = init_weights(layer_sizes[i], layer_sizes[i+1], act_name)
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    def forward(self, X):
        self.z_values = []
        self.a_values = [X]
        a = X
        for w, b, act in zip(self.weights, self.biases, self.activations):
            z = np.dot(a, w) + b
            a = act(z)
            self.z_values.append(z)
            self.a_values.append(a)
        return a

    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        output = self.a_values[-1]
        if self.last_activation == 'softmax':
            dz = (output - y) / m
        else:
            dz = (output - y) * self.act_derivs[-1](self.z_values[-1]) / m
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.a_values[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            if i != 0:
                dz = np.dot(dz, self.weights[i].T) * self.act_derivs[i-1](self.z_values[i-1])

if __name__ == "__main__":
    np.random.seed(42)
    net = FeedForwardNet(
        input_size=3,
        hidden_layers=[],
        output_size=2,
        activations=['softmax']
    )
    X = y = np.array([[0, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1],
                  [0, 1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    losses=[]
    for epoch in range(1000):
        out = net.forward(X)
        net.backward(X, y, lr=0.1)
        losses.append(-np.sum(y * np.log(out + 1e-9)) / X.shape[0])
        if epoch % 100 == 0:
            loss=-np.sum(y * np.log(out + 1e-9)) / X.shape[0]
            print(f"Epoch{epoch}, Loss:{loss:.4f}")
    print("Final Output(Probabilities):\n", net.forward(X))

    plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(0,1000,1000),losses)
    plt.show()


