import tensorflow as tf
import numpy as np
import string

if __name__ == "__main__":
    inputs = np.array([
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
         "W", "X", "Y", "Z"],
        ["Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K", "J", "I", "H", "G", "F", "E",
         "D", "C", "B", "A"],
        ["B", "D", "F", "H", "J", "L", "N", "P", "R", "T", "V", "X", "Z", "A", "C", "E", "G", "I", "K", "M", "O", "Q",
         "S", "U", "W", "Y"],
        ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H",
         "I", "J", "K", "L"],
        ["H", "G", "F", "E", "D", "C", "B", "A", "L", "K", "J", "I", "P", "O", "N", "M", "U", "T", "S", "R", "Q", "X",
         "W", "V", "Z", "Y"]
    ])

    expected = np.array([
        ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
         "X", "Y", "Z", "A"],
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
         "W", "X", "Y", "Z"],
        ["C", "E", "G", "I", "K", "M", "O", "Q", "S", "U", "W", "Y", "A", "B", "D", "F", "H", "J", "L", "N", "P", "R",
         "T", "V", "X", "Z"],
        ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H", "I",
         "J", "K", "L", "M"],
        ["I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D",
         "E", "F", "G", "H"]
    ])

    # Encode strings to int indexes
    input_encoded = np.vectorize(string.ascii_uppercase.index)(inputs)
    input_encoded = input_encoded.astype(np.float32)
    one_hot_inputs = tf.keras.utils.to_categorical(input_encoded)

    expected_encoded = np.vectorize(string.ascii_uppercase.index)(expected)
    expected_encoded = expected_encoded.astype(np.float32)
    one_hot_expected = tf.keras.utils.to_categorical(expected_encoded)

    rnn = tf.keras.layers.SimpleRNN(128, return_sequences=True)

    model = tf.keras.Sequential(
        [
            rnn,
            tf.keras.layers.Dense(len(string.ascii_uppercase)),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam")

    model.fit(one_hot_inputs, one_hot_expected, epochs=10)

    new_inputs = np.array(
        [["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Z", "A"]])
    new_inputs_encoded = np.vectorize(string.ascii_uppercase.index)(new_inputs)
    new_inputs_encoded = new_inputs_encoded.astype(np.float32)
    new_inputs_encoded = tf.keras.utils.to_categorical(new_inputs_encoded)

    # Make prediction
    prediction = model.predict(new_inputs_encoded)

    # Get prediction of last time step
    prediction = np.argmax(prediction[0][-1])
    print(prediction)
    print(string.ascii_uppercase[prediction])