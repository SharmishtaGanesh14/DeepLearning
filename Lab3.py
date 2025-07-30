import operator

def perform_operation(i1, i2, op):
    return op(i1, i2)

def compute_derivative(op, i1, i2):
    if op == operator.mul:
        return i2, i1
    elif op == operator.add:
        return 1, 1
    elif op == operator.sub:
        return 1, -1
    elif op == operator.truediv:
        return 1/i2, -i1/(i2**2)
    else:
        raise ValueError("Unsupported operation")

def node(input,weight,operation):
    output = perform_operation(input, weight, operation)
    der1,der2= compute_derivative(operation, input, weight)
    return output,der1,der2

def run():
    x1, w1, x2, w2 = 2, 3, 4, 5
    graph={"Layer1":{operator.mul:[x1,w1],operator.mul:[x2,w2]}}
    f1,der_f1x1,der_f1w1 = node(x1, w1, operator.mul)
    f2,der_f2x2,der_f2w2 = node(x2, w2, operator.mul)
    f3, der_f3f1, der_f3f2 = node(f1, f2, operator.add)

def main():
    run()
main()
