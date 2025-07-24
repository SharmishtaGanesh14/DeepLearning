import numpy as np


class Feed_forward_neural_network:
    def __init__(self, input, number_of_hidden_layers, number_of_neurons):
        self.number_of_features = len(input)
        self.input = input
        self.number_of_hidden_layers = number_of_hidden_layers
        self.number_of_neurons = number_of_neurons
        self.activation_funcs = [None] * number_of_hidden_layers

        # Initialize weights with random values instead of zeros
        self.weights_across_layers = []
        self.weights_across_layers.append(
            np.random.randn(number_of_neurons[0], self.number_of_features) * 0.01  # small random values
        )
        for i in range(1, number_of_hidden_layers):
            self.weights_across_layers.append(
                np.random.randn(number_of_neurons[i], number_of_neurons[i - 1]) * 0.01
            )

        self.bias_term = [np.zeros((neurons, 1), dtype=float) for neurons in number_of_neurons]

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
        softmax = exp_z/np.sum(np.exp(z),axis=0, keepdims=True)
        return np.array(softmax)

    def error_check(self):
        dimensions = [matrix.shape for matrix in self.weights_across_layers]

        # checking number of activation functions matches number of layers
        if len(self.activation_funcs) != len(self.weights_across_layers):
            raise ValueError(
                f"Mismatch: {len(self.activation_funcs)} activation functions for {len(self.weights_across_layers)} layers."
            )

        # checking if input features match the first weight matrix's input
        if self.number_of_features != dimensions[0][1]:
            raise ValueError(
                f"Input feature size mismatch: expected {dimensions[0][1]}, got {self.number_of_features}"
            )

        # checking if each layer's output matches next layer's input
        for i in range(len(dimensions) - 1):
            if dimensions[i][0] != dimensions[i + 1][1]:
                raise ValueError(
                    f"Layer dimension mismatch: output of layer {i + 1} is {dimensions[i][0]}, "
                    f"but input of layer {i + 2} is {dimensions[i + 1][1]}"
                )
        return None

    def user_input(self):
        # Activation function selection
        user_prompt = input("Provide custom activation functions? (Y/N): ").strip().upper()
        if user_prompt == "Y":
            valid_options = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax']
            for i in range(len(self.activation_funcs)):
                inp = input(f"Activation for layer {i + 1} (options: {valid_options}): ").strip().lower()
                if inp in valid_options:
                    if inp == "softmax" and i != len(self.activation_funcs) - 1:
                        print("Warning: Softmax usually for last layer only. Using default 'relu'")
                        self.activation_funcs[i] = "relu"
                    else:
                        self.activation_funcs[i] = inp
                else:
                    print(f"Invalid input for layer {i + 1}, using default activation.")
                    self.activation_funcs[i] = "relu" if i != len(self.activation_funcs) - 1 else "softmax"
        else:
            print("Using default activations (ReLU hidden, Softmax output).")
            for i in range(len(self.activation_funcs) - 1):
                self.activation_funcs[i] = "relu"
            self.activation_funcs[-1] = "softmax"

        # Weights input
        weights_prompt = input("Provide custom weights? (Y/N): ").strip().upper()
        if weights_prompt == "Y":
            for i, w_matrix in enumerate(self.weights_across_layers):
                print(f"Enter weights for layer {i + 1}, rows should have {w_matrix.shape[1]} comma-separated values.")
                data = input(f"Layer {i + 1} weights (rows separated by ';'): ").strip()
                try:
                    rows = data.split(';')
                    matrix = np.array([list(map(float, row.split(','))) for row in rows])
                    if matrix.shape == w_matrix.shape:
                        self.weights_across_layers[i] = matrix
                    else:
                        print(f"Shape mismatch for layer {i + 1} weights, using default random weights.")
                except:
                    print(f"Invalid input for layer {i + 1} weights, using default random weights.")
        else:
            print("Using default random weights.")

        # Bias input
        bias_prompt = input("Provide custom bias terms? (Y/N): ").strip().upper()
        if bias_prompt == "Y":
            for i, b_vector in enumerate(self.bias_term):
                data = input(f"Enter {b_vector.shape[0]} comma-separated biases for layer {i + 1}: ").strip()
                try:
                    vals = np.array(list(map(float, data.split(','))))
                    if vals.size == b_vector.shape[0]:
                        self.bias_term[i] = vals.reshape(-1, 1)
                    else:
                        print(f"Size mismatch for layer {i + 1} bias, using default zeros.")
                except:
                    print(f"Invalid input for layer {i + 1} bias, using default zeros.")
        else:
            print("Using default zero biases.")

    def run(self):
        self.error_check()
        self.user_input()
        temp = self.input.reshape(-1, 1)  # ensure input is column vector
        output_from_activation_func = {}
        i = 1
        for weight, activation_func, bias in zip(self.weights_across_layers, self.activation_funcs, self.bias_term):
            z = np.dot(weight, temp) + bias  # weighted input + bias

            if activation_func == "softmax":
                temp = self.softmax(z)
            elif activation_func == "sigmoid":
                temp = self.sigmoid(z)
            elif activation_func == "tanh":
                temp = self.tanh(z)
            elif activation_func == "relu":
                temp = self.relu(z)
            elif activation_func == "leaky_relu":
                temp = self.leaky_relu(z)
            else:
                raise ValueError(f"Unknown activation function: {activation_func}")

            output_from_activation_func[f"Layer {i}"] = temp
            i += 1

        return output_from_activation_func


def main():
    input_data = np.array([-2.4, 1.2, -0.8, 1.1])
    number_of_hidden_layers = 3
    number_of_neurons = [3, 2, 2]

    obj1 = Feed_forward_neural_network(input_data, number_of_hidden_layers, number_of_neurons)

    # Prompt user for activation functions or use default
    obj1.user_input()

    try:
        result = obj1.run()
        print(result)
    except ValueError as err:
        print(f"Caught error during run: {err}")

    print(f'The final output is {result[f"Layer {number_of_hidden_layers}"]}')


if __name__ == "__main__":
    main()
