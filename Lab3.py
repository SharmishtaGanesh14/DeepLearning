class Node:
    def __init__(self, inputs=None, value=None):
        self.value = value
        self.grad = 0.0
        self.inputs = inputs or []

class InputNode(Node):
    def forward(self, val=None):
        if val is not None:
            self.value = val
        return self.value

    def backward(self, grad_op):
        self.grad += grad_op

class AddNode(Node):
    def forward(self):
        self.value = sum(node.forward() for node in self.inputs)
        return self.value

    def backward(self, grad_op):
        for node in self.inputs:
            node.backward(grad_op)

class SubNode(Node):
    def forward(self):
        if not self.inputs:
            self.value = 0.0
            return self.value
        diff = self.inputs[0].forward()
        for node in self.inputs[1:]:
            diff -= node.forward()
        self.value = diff
        return self.value

    def backward(self, grad_op):
        if not self.inputs:
            return
        self.inputs[0].backward(grad_op)
        for node in self.inputs[1:]:
            node.backward(-grad_op)

class MulNode(Node):
    def forward(self):
        if len(self.inputs) != 2:
            raise ValueError("MulNode requires exactly 2 inputs")
        a = self.inputs[0].forward()
        b = self.inputs[1].forward()
        self.value = a * b
        return self.value

    def backward(self, grad_op):
        if len(self.inputs) != 2:
            return
        x, y = self.inputs
        x.backward(grad_op * y.value)
        y.backward(grad_op * x.value)

# def reset_gradients(node):
#     node.grad = 0.0
#     for inp in node.inputs:
#         reset_gradients(inp)

def main():
    x1 = InputNode()
    w1 = InputNode()
    x2 = InputNode()
    w2 = InputNode()
    mulN = MulNode(inputs=[x1, w1])
    subN = SubNode(inputs=[x2, w2])
    addN = AddNode(inputs=[mulN, subN])
    x1.forward(3.0)
    w1.forward(2.0)
    x2.forward(4.0)
    w2.forward(5.0)
    res = addN.forward()
    print(res)
    # reset_gradients(addN)
    addN.backward(1.0)
    print("dz/dx1:", x1.grad)
    print("dz/dw1:", w1.grad)
    print("dz/dx2:", x2.grad)
    print("dz/dw2:", w2.grad)

if __name__ == "__main__":
    main()
