# Date created: 16/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Implement various update rules used to optimize the neural network

import torch

# Base Optimizer
class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# SGD
class SGD(Optimizer):
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            p.data -= self.lr * p.grad

# SGD + Momentum
class SGDMomentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
            p.data += self.velocity[i]

# AdaGrad
class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-8):
        super().__init__(params, lr)
        self.eps = eps
        self.historical_grad = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.historical_grad[i] += p.grad ** 2
            adjusted_lr = self.lr / (torch.sqrt(self.historical_grad[i]) + self.eps)
            p.data -= adjusted_lr * p.grad

# RMSProp
class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(params, lr)
        self.beta = beta
        self.eps = eps
        self.sq_grad_avg = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.sq_grad_avg[i] = self.beta * self.sq_grad_avg[i] + (1 - self.beta) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (torch.sqrt(self.sq_grad_avg[i]) + self.eps)

# Adam
class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue

            # Update biased first moment (m) and second moment (v)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update params
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


# Dummy dataset
X = torch.randn(20, 10)
y = torch.randn(20, 1)

# Simple linear model
model = torch.nn.Linear(10, 1)
criterion = torch.nn.MSELoss()

# Choose optimizer
# optimizer = SGD(model.parameters(), lr=0.1)
# optimizer = SGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
# optimizer = AdaGrad(model.parameters(), lr=0.1)
# optimizer = RMSProp(model.parameters(), lr=0.01)
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
