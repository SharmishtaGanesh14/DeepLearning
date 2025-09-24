import numpy as np

def RNN(hidden_state, inp, output_dim):
    xh, yh = hidden_state.shape # hidden state shape
    xi, yi = inp.shape # input shape (rows, cols)
    xo, yo = output_dim.shape # output dimensions

    # # Initialize weight matrices
    # Wxh = [[0 for _ in range(xi)] for _ in range(xh)]   # input → hidden
    # Whh = [[0 for _ in range(xh)] for _ in range(xh)]   # hidden → hidden
    # Why = [[0 for _ in range(xh)] for _ in range(xo)]   # hidden → output
    #
    # # Fill Wxh
    # print("\nEnter weights for Wxh (input → hidden):")
    # for i in range(xh):
    #     for j in range(xi):
    #         num = float(input(f"Wxh[{i},{j}] = "))
    #         Wxh[i][j] = num
    #
    # # Fill Whh
    # print("\nEnter weights for Whh (hidden → hidden):")
    # for i in range(xh):
    #     for j in range(xh):
    #         num = float(input(f"Whh[{i},{j}] = "))
    #         Whh[i][j] = num
    #
    # # Fill Why
    # print("\nEnter weights for Why (hidden → output):")
    # for i in range(xo):
    #     for j in range(xh):
    #         num = float(input(f"Why[{i},{j}] = "))
    #         Why[i][j] = num

    Wxh=[[0.5,-0.3],[0.8,0.2],[0.1,0.4]]
    Whh=[[0.1,0.4,0],[-0.2,0.3,0.2],[0.05,-0.1,0.2]]
    Why=[[1,-1,0.5],[0.5,0.5,-0.5]]

    Wxh = np.array(Wxh)
    Whh = np.array(Whh)
    Why = np.array(Why)

    print("\nWxh (input → hidden):\n", Wxh)
    print("\nWhh (hidden → hidden):\n", Whh)
    print("\nWhy (hidden → output):\n", Why)

    outputs=[]
    hidden_op=[]
    hidden_op.append(hidden_state)
    for i in range(xi):
        input_vector=inp[i].reshape(-1,1)
        hid=np.tanh(np.dot(Wxh, input_vector)+np.dot(Whh, hidden_state))
        hidden_state=hid
        y=np.dot(Why, hidden_state)
        outputs.append(y)
        hidden_op.append(hidden_state)
    return outputs, hidden_op


if __name__ == '__main__':
    hidden_state = np.array([[0, 0, 0]]).T # (3×1 hidden vector)
    inp = np.array([[1, 2], [-1, 1]])  # (2×2 input matrix)
    output_dim = np.array([[2, 1]]).T  # (2×1 output vector)

    ops,hidden_states = RNN(hidden_state, inp, output_dim)
    print(ops)
    print(hidden_states)