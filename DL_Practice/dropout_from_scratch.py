import torch
import numpy as np

class Dropout:
    def __init__(self, dropout=0.5):
        self.dropout = dropout
        self.mask = None

    def forward(self, x, train=True):
        if train:
            dim1, dim2 = x.size()
            # create mask with 0s and 1s
            self.mask = (torch.rand(dim1, dim2) > self.dropout).float()
            # scale during training
            return (x * self.mask) / (1 - self.dropout)
        else:
            # during inference, no dropout applied
            return x

    def backward(self, gradient_op):
        # pass gradient only where mask=1
        return (gradient_op * self.mask) / (1 - self.dropout)


# Example
input = torch.randn(10, 1)
gradient_op = torch.ones_like(input)

dropout = Dropout(0.5)

out_train = dropout.forward(input, train=True)
out_test = dropout.forward(input, train=False)
grad_back = dropout.backward(gradient_op)

print("Input:\n", input)
print("Output (train):\n", out_train)
print("Output (test):\n", out_test)
print("Backward gradient:\n", grad_back)
