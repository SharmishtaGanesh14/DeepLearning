import numpy as np

def apply_padding_striding(matrix, padding_size=1, striding_size=1):
    rows, cols = matrix.shape
    # Pad left and right (columns)
    left_pad = np.zeros((rows, padding_size))
    right_pad = np.zeros((rows, padding_size))
    matrix = np.hstack([left_pad, matrix, right_pad])
    # Pad top and bottom (rows)
    new_cols = matrix.shape[1]
    top_pad = np.zeros((padding_size, new_cols))
    bottom_pad = np.zeros((padding_size, new_cols))
    matrix = np.vstack([top_pad, matrix, bottom_pad])
    return matrix/striding_size

def convolution_operation(input,filter):
    op=[]
    inp_shape=input.shape[0]
    filter_shape=filter.shape[0]
    i=0
    j=0
    while i < -filter_shape+inp_shape+1:
        j=0
        while j < -filter_shape+inp_shape+1:
            op.append((input[i:i+filter_shape,:][:,j:j+filter_shape]*filter).sum())
            j+=1
        i += 1
    op=np.array(op).reshape(inp_shape-filter_shape+1,inp_shape-filter_shape+1)
    op=apply_padding_striding(op)
    return op

image1=np.array([[3,0,1,2,7,4],
        [1,5,8,9,3,1],
        [2,7,2,5,1,3],
        [0,1,3,1,7,8],
        [4,2,1,6,2,8],
        [2,4,5,2,3,9]])
filter=np.array([[1,0,-1],
        [1,0,-1],
        [1,0,-1]])
print(convolution_operation(image1,filter))
