# Date created: 16/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Implement batch normalization from scratch. and layer normalization for training deep networks.

import numpy as np
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9, lr=0.01):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.eps = eps
        self.lr = lr
        self.momentum = momentum
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, x, training=True):
        self.x = x
        if training:
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            out = self.gamma * self.x_norm + self.beta

            # update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, grad_op):
        N = self.x.shape[0]

        # Gradients wrt gamma and beta
        d_gamma = np.sum(grad_op * self.x_norm, axis=0, keepdims=True)
        d_beta = np.sum(grad_op, axis=0, keepdims=True)

        # Update params
        self.gamma -= self.lr * d_gamma
        self.beta -= self.lr * d_beta

        # Grad wrt input
        dx_norm = grad_op * self.gamma
        d_var = np.sum(dx_norm * (self.x - self.batch_mean) * -0.5 * (self.batch_var + self.eps) ** (-1.5),
                       axis=0, keepdims=True)
        d_mean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.eps), axis=0, keepdims=True) + \
                 d_var * np.mean(-2 * (self.x - self.batch_mean), axis=0, keepdims=True)
        dx = dx_norm / np.sqrt(self.batch_var + self.eps) + \
             d_var * 2 * (self.x - self.batch_mean) / N + \
             d_mean / N

        return dx

class LayerNorm:
    def __init__(self, num_features, eps=1e-5, lr=0.01):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.eps = eps
        self.lr = lr

    def forward(self, x):
        self.x = x
        # mean and var across features (axis=1)
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, grad_op):
        N, D = self.x.shape

        # Gradients wrt gamma and beta
        d_gamma = np.sum(grad_op * self.x_norm, axis=0, keepdims=True)
        d_beta = np.sum(grad_op, axis=0, keepdims=True)

        # Update params
        self.gamma -= self.lr * d_gamma
        self.beta -= self.lr * d_beta

        # Grad wrt input
        dx_norm = grad_op * self.gamma
        d_var = np.sum(dx_norm * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=1, keepdims=True)
        d_mean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=1, keepdims=True) + \
                 d_var * np.mean(-2 * (self.x - self.mean), axis=1, keepdims=True)
        dx = dx_norm / np.sqrt(self.var + self.eps) + d_var * 2 * (self.x - self.mean) / D + d_mean / D

        return dx

# Toy dataset: 4 samples, 3 features
X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0]
])
# Fake gradient coming from next layer (same shape as X)
d_out = np.array([
    [0.1, 0.2, 0.3],
    [0.2, 0.1, 0.4],
    [0.3, 0.4, 0.1],
    [0.5, 0.2, 0.2]
])

# Initialize BN
bn = BatchNorm(num_features=3, lr=0.01)
# Forward pass
out = bn.forward(X, training=True)
print("Forward Output:\n", out)
# Backward pass
dx = bn.backward(d_out)
print("\nGrad wrt input dx:\n", dx[0])   # input gradient
print("\nUpdated gamma:\n", bn.gamma)
print("Updated beta:\n", bn.beta)

# Initialise LN
ln = LayerNorm(num_features=3, lr=0.01)
out = ln.forward(X)
print("\nForward Output:\n", out)
dx = ln.backward(d_out)
print("\nGrad wrt input dx:\n", dx)
print("\nUpdated gamma:\n", ln.gamma)
print("Updated beta:\n", ln.beta)