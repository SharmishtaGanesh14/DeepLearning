# Date created: 22/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Implement convolution operations from scratch. Assume a 3x3 kernel and apply it on an input image of 32x32.
# Implement maxpool operation from scratch.

import numpy as np

def apply_padding(matrix, padding_size=1):
    rows, cols = matrix.shape
    left_pad = np.zeros((rows, padding_size))
    right_pad = np.zeros((rows, padding_size))
    matrix = np.hstack([left_pad, matrix, right_pad])
    new_cols = matrix.shape[1]
    top_pad = np.zeros((padding_size, new_cols))
    bottom_pad = np.zeros((padding_size, new_cols))
    matrix = np.vstack([top_pad, matrix, bottom_pad])
    return matrix

def convolution_operation(input, filter, padding=1, stride=1):
    input = apply_padding(input, padding)
    inp_shape = input.shape[0]
    filter_shape = filter.shape[0]
    op_shape = ((inp_shape - filter_shape) // stride) + 1
    op = []
    for i in range(0, op_shape * stride, stride):
        for j in range(0, op_shape * stride, stride):
            patch = input[i:i + filter_shape, j:j + filter_shape]
            op.append((patch * filter).sum())
    op = np.array(op).reshape(op_shape, op_shape)
    return op

def max_pooling_operation(input, filter_shape, stride=1):
    inp_shape = input.shape[0]
    op_shape = (inp_shape - filter_shape) // stride + 1
    op = []
    for i in range(0, op_shape * stride, stride):
        for j in range(0, op_shape * stride, stride):
            patch = input[i:i + filter_shape, j:j + filter_shape]
            op.append(patch.max())
    op = np.array(op).reshape(op_shape, op_shape)
    return op

if __name__ == '__main__':
    np.random.seed(0)
    image1 = np.random.randint(0,255,size=(32,32))
    filter = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
    op = convolution_operation(image1, filter, padding=1, stride=2)
    print("Convolution output:\n", op)
    maxpool_filter_size=3
    op2 = max_pooling_operation(op, maxpool_filter_size, stride=1)
    print("Max pooling output:\n", op2)
