# Date created: 16/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Implement dropout from scratch and layer normalization for training deep networks

import numpy as np
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, x,training=True):
        if training:
            dim1,dim2=x.shape
            self.mask=(np.random.rand(dim1,dim2)>self.p).astype(np.float32)
            return x*self.mask/(1-self.p)
        else:
            return x

    def backward(self, grad_output):
        return grad_output*self.mask/(1-self.p)

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
grad_op=np.ones_like(X)

dropout = Dropout(p=0.5)
# Training mode
out_train = dropout.forward(X, training=True)
print("Forward (training):\n", out_train)
# Inference mode
out_test = dropout.forward(X, training=False)
print("\nForward (inference):\n", out_test)
# Backward pass
dx = dropout.backward(grad_op)
print("\nBackward gradient:\n", dx)





