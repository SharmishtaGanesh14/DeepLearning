import numpy as np
import string
from typing import List


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def string_to_one_hot(inputs: np.ndarray) -> np.ndarray:
    char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}

    one_hot_inputs = []
    for row in inputs:
        one_hot_list = []
        for char in row:
            if char.upper() in char_to_index:
                one_hot_vector = np.zeros((len(string.ascii_uppercase), 1))
                one_hot_vector[char_to_index[char.upper()]] = 1
                one_hot_list.append(one_hot_vector)
        one_hot_inputs.append(np.array(one_hot_list))
    return np.array(one_hot_inputs)


class InputLayer:
    def __init__(self, inputs: np.ndarray, hidden_size: int) -> None:
        self.inputs = inputs
        self.U = np.random.uniform(low=0, high=1, size=(hidden_size, len(string.ascii_uppercase)))
        self.delta_U = np.zeros_like(self.U)

    def weighted_sum(self, time_step: int) -> np.ndarray:
        return self.U @ self.inputs[time_step]

    def calculate_deltas_per_step(self, time_step: int, delta_hidden: np.ndarray) -> None:
        self.delta_U += delta_hidden @ self.inputs[time_step].T

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.U -= learning_rate * self.delta_U
        self.delta_U.fill(0.0)


class HiddenLayer:
    def __init__(self, vocab_size: int, size: int) -> None:
        self.W = np.random.uniform(low=0, high=1, size=(size, size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(vocab_size, size, 1))
        self.next_delta_activation = np.zeros(shape=(size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_W = np.zeros_like(self.W)

    def get_hidden_state(self, time_step: int) -> np.ndarray:
        if time_step < 0:
            return np.zeros_like(self.states[0])
        return self.states[time_step]

    def set_hidden_state(self, time_step: int, hidden_state: np.ndarray) -> None:
        self.states[time_step] = hidden_state

    def activate(self, weighted_input: np.ndarray, time_step: int) -> np.ndarray:
        previous_hidden_state = self.get_hidden_state(time_step - 1)
        weighted_hidden_state = self.W @ previous_hidden_state
        weighted_sum = weighted_input + weighted_hidden_state + self.bias
        activation = np.tanh(weighted_sum)
        self.set_hidden_state(time_step, activation)
        return activation

    def calculate_deltas_per_step(self, time_step: int, delta_output: np.ndarray) -> np.ndarray:
        delta_activation = delta_output + self.next_delta_activation
        delta_weighted_sum = delta_activation * (1 - self.get_hidden_state(time_step) ** 2)
        self.next_delta_activation = self.W.T @ delta_weighted_sum
        self.delta_W += delta_weighted_sum @ self.get_hidden_state(time_step - 1).T
        self.delta_bias += delta_weighted_sum
        return delta_weighted_sum

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.W -= learning_rate * self.delta_W
        self.bias -= learning_rate * self.delta_bias
        self.delta_W.fill(0.0)
        self.delta_bias.fill(0.0)


class OutputLayer:
    def __init__(self, size: int, hidden_size: int) -> None:
        self.V = np.random.uniform(low=0, high=1, size=(size, hidden_size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = []
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_V = np.zeros_like(self.V)

    def predict(self, hidden_state: np.ndarray, time_step: int) -> np.ndarray:
        output = self.V @ hidden_state + self.bias
        prediction = softmax(output)
        if len(self.states) <= time_step:
            self.states.append(prediction)
        else:
            self.states[time_step] = prediction
        return prediction

    def get_state(self, time_step: int) -> np.ndarray:
        return self.states[time_step]

    def calculate_deltas_per_step(self, expected: np.ndarray, hidden_state: np.ndarray, time_step: int) -> np.ndarray:
        delta_output = self.get_state(time_step) - expected
        self.delta_V += delta_output @ hidden_state.T
        self.delta_bias += delta_output
        return self.V.T @ delta_output

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.V -= learning_rate * self.delta_V
        self.bias -= learning_rate * self.delta_bias
        self.delta_V.fill(0.0)
        self.delta_bias.fill(0.0)


class VanillaRNN:
    def __init__(self, vocab_size: int, hidden_size: int, alpha: float) -> None:
        self.hidden_layer = HiddenLayer(vocab_size, hidden_size)
        self.output_layer = OutputLayer(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.input_layer = None

    def feed_forward(self, inputs: np.ndarray) -> OutputLayer:
        self.input_layer = InputLayer(inputs, self.hidden_size)
        self.output_layer.states = []
        for step in range(len(inputs)):
            weighted_input = self.input_layer.weighted_sum(step)
            activation = self.hidden_layer.activate(weighted_input, step)
            self.output_layer.predict(activation, step)
        return self.output_layer

    def backpropagation(self, expected: np.ndarray) -> None:
        for step_number in reversed(range(len(expected))):
            delta_output = self.output_layer.calculate_deltas_per_step(
                expected[step_number],
                self.hidden_layer.get_hidden_state(step_number),
                step_number,
            )
            delta_weighted_sum = self.hidden_layer.calculate_deltas_per_step(step_number, delta_output)
            self.input_layer.calculate_deltas_per_step(step_number, delta_weighted_sum)

        self.output_layer.update_weights_and_bias(self.alpha)
        self.hidden_layer.update_weights_and_bias(self.alpha)
        self.input_layer.update_weights_and_bias(self.alpha)

    def loss(self, y_hat: List[np.ndarray], y: List[np.ndarray]) -> float:
        return sum(-np.sum(y[i] * np.log(y_hat[i])) for i in range(len(y)))

    def train(self, inputs: np.ndarray, expected: np.ndarray, epochs: int) -> None:
        for epoch in range(epochs):
            print(f"epoch={epoch}")
            for idx, input in enumerate(inputs):
                y_hats = self.feed_forward(input)
                self.backpropagation(expected[idx])
                print(f"Loss round: {self.loss([y for y in y_hats.states], expected[idx])}")


if __name__ == "__main__":
    inputs = np.array([
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        list("ZYXWVUTSRQPONMLKJIHGFEDCBA"),
        list("BDFHJLNPRTVXZACEGIKMOQSUWY"),
        list("MNOPQRSTUVWXYZABCDEFGHIJKL"),
        list("HGFEDCBALKJIPONMUTSRQXWVZY")
    ])

    expected = np.array([
        list("BCDEFGHIJKLMNOPQRSTUVWXYZA"),
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        list("CEGIKMOQSUWYABDFHJLNPRTVXZ"),
        list("NOPQRSTUVWXYZABCDEFGHIJKLM"),
        list("IJKLMNOPQRSTUVWXYZABCDEFGH")
    ])

    one_hot_inputs = string_to_one_hot(inputs)
    one_hot_expected = string_to_one_hot(expected)

    rnn = VanillaRNN(vocab_size=len(string.ascii_uppercase), hidden_size=128, alpha=0.0001)
    rnn.train(one_hot_inputs, one_hot_expected, epochs=10)

    new_inputs = np.array([["B", "C", "D"]])
    for input in string_to_one_hot(new_inputs):
        predictions = rnn.feed_forward(input)
        output = np.argmax(predictions.states[-1])
        print(output)  # index
        print(string.ascii_uppercase[output])  # predicted character
