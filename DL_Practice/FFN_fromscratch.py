import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability trick
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def identity(x):
    return x

def FFN(input, output_size, hidden_sizes=None, activation_fn='relu', final_activation=None):
    if hidden_sizes is None:
        hidden_sizes = []

    input_size = input.shape[0]
    layers = [input_size] + hidden_sizes + [output_size]

    # init weights & biases
    weights, biases = [], []
    for i in range(1, len(layers)):
        weights.append(np.random.randn(layers[i], layers[i-1]))
        biases.append(np.random.randn(layers[i], 1))

    # activation mapping
    activation_map = {'relu': relu, 'sigmoid': sigmoid, 'softmax': softmax, None: identity}
    if activation_fn not in activation_map or final_activation not in activation_map:
        raise ValueError("Invalid activation")

    activation_fn = activation_map[activation_fn]
    final_activation = activation_map[final_activation]

    # forward pass
    a = input
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = W @ a + b
        if i == len(layers) - 2:   # last layer
            a = final_activation(z)
        else:
            a = activation_fn(z)

    return a

input_vec = np.random.randn(4, 1).astype(np.float32)

# 4 → [3,2] → 1 with ReLU hidden and identity output
print("Network 1 output:", FFN(input_vec, output_size=1, hidden_sizes=[], activation_fn="relu"))

# 4 → [3,2] → 2 with ReLU hidden and softmax output
print("Network 2 output:", FFN(input_vec, output_size=2, hidden_sizes=[3,2], activation_fn="relu", final_activation="softmax"))
