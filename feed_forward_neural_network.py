import numpy as np
class Feed_forward_neural_network:
    def __init__(self, number_of_features,input, number_of_hidden_layers, number_of_neurons, weights_across_layers,
                 activation_funcs):
        self.number_of_features = number_of_features
        self.input=input
        self.number_of_hidden_layers = number_of_hidden_layers
        self.number_of_neurons = number_of_neurons
        self.weights_across_layers = weights_across_layers
        self.activation_funcs = activation_funcs

    def sigmoid(self,z):
        sig_z = 1 / (1 + np.exp(-z))
        return np.array(sig_z)

    def tanh(self,z):
        tanh_z = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return np.array(tanh_z)

    def relu(self,z):
        ReLU = (z + np.abs(z)) / 2
        return np.array(ReLU)

    def leaky_relu(self,z):
        leaky_ReLU = np.maximum(0.01 * z, z)
        return np.array(leaky_ReLU)

    def softmax(self,z):
        denom = sum(np.exp(val) for val in z)
        num = np.exp(z)
        softmax = num / denom
        return np.array(softmax)

    def error_check(self):
        dimensions = [matrix.shape for matrix in self.weights_across_layers]

        # checking number of activation functions matches number of layers
        if len(self.activation_funcs) != len(self.weights_across_layers):
            raise ValueError(
                f"Mismatch: {len(self.activation_funcs)} activation functions for {len(self.weights_across_layers)} layers."
            )

        # checking if input features match the first weight matrix's input
        if self.number_of_features != dimensions[0][0]:
            raise ValueError(
                f"Input feature size mismatch: expected {dimensions[0][0]}, got {self.number_of_features}"
            )

        # checking if each layer's output matches next layer's input
        for i in range(len(dimensions) - 1):
            if dimensions[i][1] != dimensions[i + 1][0]:
                raise ValueError(
                    f"Layer dimension mismatch: output of layer {i + 1} is {dimensions[i][1]}, "
                    f"but input of layer {i + 2} is {dimensions[i + 1][0]}"
                )
        return None

    def run(self):
        # MY LOGIC
        self.error_check()
        temp = self.input
        output_from_activation_func = {}
        i=1
        for weight, activation_func in zip(self.weights_across_layers, self.activation_funcs):
            temp2 = []
            temp = np.dot(temp, weight)
            if activation_func=="softmax":
                temp2=self.softmax(temp)
            for op in temp:
                if activation_func == "sigmoid":
                    activated = self.sigmoid(op)
                elif activation_func == "tanh":
                    activated = self.tanh(op)
                elif activation_func == "relu":
                    activated = self.relu(op)
                elif activation_func == "leaky_relu":
                    activated = self.leaky_relu(op)
                elif activation_func=="softmax":
                    break
                else:
                    raise ValueError(f"Unknown activation function: {activation_func}")
                temp2.append(activated)
            output_from_activation_func[f"Layer {i}"]=temp2
            temp = np.array(output_from_activation_func[f"Layer {i}"])
            i=i+1
        return output_from_activation_func

        # # SIMPLIFIED
        #
        # temp = self.input
        # for weight, activation_func in zip(self.weights_across_layers, self.activation_funcs):
        #     temp = np.dot(temp, weight)
        #     if activation_func == "softmax":
        #         temp = self.softmax(temp)
        #     else:
        #         temp = np.array([getattr(self, activation_func)(z) for z in temp])
        # return temp

number_of_features=4
input=np.array([-2.4,1.2,-0.8,1.1])
number_of_hidden_layers=3
number_of_neurons=[3,2,2]
weights_across_layers = np.array([
    np.array([  # Layer 1 weights: shape (4 inputs, 3 neurons)
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1]
    ]),
    np.array([  # Layer 2 weights: shape (3 inputs, 2 neurons)
        [0.001, 0.001],
        [0.001, 0.001],
        [0.001, 0.001]
    ]),
    np.array([  # Layer 3 (output layer): shape (2 inputs, 2 neurons)
        [0.01, 0.01],
        [0.01, 0.01]
    ])
],dtype=object)
activation_funcs=np.array(["relu","relu","softmax"],dtype=object)
obj1=Feed_forward_neural_network(number_of_features,input,number_of_hidden_layers,number_of_neurons,weights_across_layers,activation_funcs)
try:
    result = obj1.run()
    print(result)
except ValueError as err:
    print(f"Caught error during run: {err}")
print(f'The final output is {result[f"Layer {number_of_hidden_layers}"]}')