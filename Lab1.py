import numpy as np
import matplotlib.pyplot as plt

# a) Sigmoid
def sigmoid(z):
    sig_z = 1 / (1 + np.exp(-z))
    return np.array(sig_z), np.array(sig_z * (1 - sig_z)),np.mean(sig_z)

# b) Tanh
def tanh(z):
    tanh_z = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return np.array(tanh_z), np.array(1 - tanh_z ** 2),np.mean(tanh_z)

# c) ReLU
def relu(z):
    ReLU = (z + np.abs(z)) / 2
    der_ReLU = 0.5 + 0.5 * np.sign(z)
    return np.array(ReLU), np.array(der_ReLU),np.mean(ReLU)

# d) Leaky ReLU
def leaky_relu(z):
    leaky_ReLU = np.maximum(0.01 * z, z)
    der_leaky_ReLU = np.where(z > 0, 1, 0.01)
    return np.array(leaky_ReLU), np.array(der_leaky_ReLU),np.mean(leaky_ReLU)

# e) Softmax and its Jacobian
def softmax(z):
    K = len(z)
    denom = sum(np.exp(val) for val in z)
    num = np.exp(z)
    softmax = num / denom
    jacobian = []
    for i in range(len(z)):
        temp = []
        for j in range(len(z)):
            if i == j:
                temp.append(softmax[i] * (1 - softmax[i]))
            else:
                temp.append(-softmax[i] * softmax[j])
        jacobian.append(temp)
    return np.array(softmax), np.array(jacobian),np.mean(softmax)

def plot_activation(z, func_name, func):
    y, dy, mean = func(z)
    print(f'Mean of the {func_name} output: {mean}')
    plt.figure(figsize=(8, 6))
    plt.plot(z, y, label=f"{func_name} function")
    plt.plot(z, dy, label=f"Derivative of {func_name}")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    z = np.linspace(-10, 10, 100)

    plot_activation(z, "Sigmoid", sigmoid)
    plot_activation(z, "Tanh", tanh)
    plot_activation(z, "ReLU", relu)
    plot_activation(z, "Leaky ReLU", leaky_relu)

    sft, derv_sft, mean_sft = softmax(z)
    print("Mean of softmax: ",mean_sft)
    plt.figure(figsize=(8, 6))
    plt.plot(z, sft, label="Softmax function")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
    print("Softmax output:", sft)
    print("Softmax Jacobian:")
    print(derv_sft)


# Questions

# Sigmoid
# a) The minimum value is 0 and maximum value is 1
# b) No, the values are not zero centered, its centered around 0.5
# c) Smooth gradient for small inputs, flattens for large positive/negative values.
# - sigmoid is generally not used because of this, gradient becomes zero for very small or very large values,
# if gradient becomes zero then there is no update/learning thats happening then it becomes redundant

# Tanh
# a) The minimum value is -1 and maximum value is 1
# b) Yes, the values are zero centered
# c) Smooth gradient for small inputs, flattens for large positive/negative values.

# ReLU - normally used
# a) The minimum value is 0 and maximum value is inf (Linear growth for positive inputs)
# b) No, the values are not zero centered, the values are always more than 0
# c) Gradient:  For positive values: gradient = 1
#    Gradient:  For negative values: gradient = 0
# cost of computing maximum is less than computing exponential
# - so this is better and computationally less expensive than other activation func
# Always keep in mind
# - time
# - space
# - computational power

# Leaky ReLU
# a) The minimum value is -0.01 and maximum value is inf (Linear growth for positive inputs)
# b) No, the values are not zero centered
# c) Gradient:  For positive values: gradient = 1
#    Gradient:  For negative values: gradient = 0.01

# Softmax - computationally expensive
# a) The minimum value is 0 and maximum value is 1 (exponential growth for positive inputs)
# b) No, the values are not zero centered

# Relationship bw softmax and tanh
# tanh(z)=2*sigmoid(2z)-1

# zero centered = average of all function outputs will be zero