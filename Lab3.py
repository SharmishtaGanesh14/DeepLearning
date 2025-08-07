import numpy as np
class Node():
    def __init__(self, input=[], value=None):
        self.value = value
        self.grad = 0
        self.input = input
        self.output = []
        for node in self.input:
            node.output.append(self)


class InputNode(Node):
    def forward(self):
        return self.value

    def backward(self, grad_op):
        self.grad += grad_op
        return self.grad


class AddNode(Node):
    def forward(self):
        self.value = sum(node.forward() for node in self.input)
        return self.value

    def backward(self, grad_op):
        for node in self.input:
            node.backward(grad_op)


class SubNode(Node):
    def forward(self):
        diff = self.input[0].forward()
        for node in self.input[1:]:
            diff -= node.forward()
        self.value = diff
        return self.value

    def backward(self, grad_op):
        self.input[0].backward(grad_op)
        for node in self.input[1:]:
            node.backward(-grad_op)


class MulNode(Node):
    def forward(self):
        a = self.input[0].forward()
        b = self.input[1].forward()
        self.value = a * b
        return self.value

    def backward(self, grad_op):
        x, y = self.input
        x.backward(grad_op * y.value)
        y.backward(grad_op * x.value)


class ReluNode(Node):
    def forward(self):
        input_val = self.input[0].forward()  # Call forward on input
        self.value = max(0, input_val)
        return self.value

    def backward(self, grad_op):
        input_node = self.input[0]
        if input_node.value > 0:
            input_node.backward(grad_op)
        else:
            input_node.backward(0)


class SigmoidNode(Node):
    def forward(self):
        input_val = self.input[0].forward()  # Call forward on input
        self.value = 1 / (1 + np.exp(-input_val))
        return self.value

    def backward(self, grad_op):
        input_node = self.input[0]
        sig = self.value
        input_node.backward(grad_op * sig * (1 - sig))


def reset_gradients(node):
    node.grad = 0.0
    for inp in node.input:
        reset_gradients(inp)


def main():
    # 1)
    x1 = InputNode(value=3)
    w1 = InputNode(value=2)
    x2 = InputNode(value=4)
    w2 = InputNode(value=5)
    mulN = MulNode(input=[x1, w1])
    subN = SubNode(input=[x2, w2])
    addN = AddNode(input=[mulN, subN])
    x1.forward()
    w1.forward()
    x2.forward()
    w2.forward()
    res3 = addN.forward()
    print(res3)
    # reset_gradients(addN)
    addN.backward(1.0)
    print("dz/dx1:", x1.grad)
    print("dz/dw1:", w1.grad)
    print("dz/dx2:", x2.grad)
    print("dz/dw2:", w2.grad)

    # 2
    w1 = InputNode(value=0.5)
    w2 = InputNode(value=0.5)
    w3 = InputNode(value=0.5)
    x1= InputNode(value=1)
    inp1=MulNode(input=[x1, w1])
    sigNode=SigmoidNode(input=[inp1])
    inp2 = MulNode(input=[sigNode, w2])
    sigNode2=SigmoidNode(input=[inp2])
    inp3 = MulNode(input=[sigNode2, w3])
    sigNode3=SigmoidNode(input=[inp3])
    x1.forward()
    print(sigNode3.forward())
    sigNode3.backward(1)
    print(x1.grad)


if __name__ == "__main__":
    main()