# Date created: 22/08/2025
# Date updated: 30/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Generalized convolution and maxpooling implementation from scratch

import numpy as np

def apply_padding(matrix, padding_size=1):
    """Apply zero padding to a 2D matrix."""
    rows, cols = matrix.shape
    left_pad = np.zeros((rows, padding_size))
    right_pad = np.zeros((rows, padding_size))
    matrix = np.hstack([left_pad, matrix, right_pad])
    new_cols = matrix.shape[1]
    top_pad = np.zeros((padding_size, new_cols))
    bottom_pad = np.zeros((padding_size, new_cols))
    matrix = np.vstack([top_pad, matrix, bottom_pad])
    return matrix

def convolution_operation(input, filters, padding=1, stride=1):
    """
    Perform convolution on input with multiple filters.
    input   : shape (C, H, W)
    filters : shape (N, C, k, k)  (N filters, C channels, kxk kernel)
    Returns : shape (N, H_out, W_out)
    """
    C, H, W = input.shape
    N, _, k, _ = filters.shape

    # Apply padding per channel
    padded = np.array([apply_padding(input[c], padding) for c in range(C)])
    H_p, W_p = padded.shape[1], padded.shape[2]

    # Output dimensions
    H_out = (H_p - k) // stride + 1
    W_out = (W_p - k) // stride + 1

    output = np.zeros((N, H_out, W_out))

    for n in range(N):  # loop over filters
        for i in range(0, H_out * stride, stride):
            for j in range(0, W_out * stride, stride):
                patch = padded[:, i:i+k, j:j+k]   # shape (C, k, k)
                output[n, i//stride, j//stride] = np.sum(patch * filters[n])
    return output

def max_pooling_operation(input, filter_size=2, stride=2):
    """
    Perform max pooling on input.
    input : shape (C, H, W)
    filter_size : pooling kernel size (int)
    stride : pooling stride
    Returns : shape (C, H_out, W_out)
    """
    C, H, W = input.shape
    H_out = (H - filter_size) // stride + 1
    W_out = (W - filter_size) // stride + 1
    output = np.zeros((C, H_out, W_out))

    for c in range(C):
        for i in range(0, H_out * stride, stride):
            for j in range(0, W_out * stride, stride):
                patch = input[c, i:i+filter_size, j:j+filter_size]
                output[c, i//stride, j//stride] = np.max(patch)
    return output

if __name__ == '__main__':
    np.random.seed(0)
    image = np.random.randint(0, 255, size=(3, 32, 32))  # (C,H,W)

    # Define 2 filters of size 3x3 for 3 channels
    filters = np.array([
        [[[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]],

         [[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]],

         [[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]]],

        [[[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]],

         [[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]],

         [[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]]]
    ])

    conv_out = convolution_operation(image, filters, padding=1, stride=1)
    print("Convolution output shape:", conv_out.shape)

    pooled_out = max_pooling_operation(conv_out, filter_size=2, stride=2)
    print("Max pooling output shape:", pooled_out.shape)
