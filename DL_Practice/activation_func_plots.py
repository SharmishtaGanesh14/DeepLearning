import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_of_sigmoid(x):
    y=sigmoid(x)
    return y*(1-y)

def tanh(x):
    return np.tanh(x)

def derivative_of_tanh(x):
    y=tanh(x)
    return 1-y**2

def relu(x):
    return np.maximum(0,x)

def derivative_of_relu(x):
    return np.where(x>0,1,0)

def leaky_relu(x,alpha=0.01):
    return np.maximum(alpha*x,x)

def derivative_of_leaky_relu(x,alpha=0.01):
    return np.where(x>0,1,alpha)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def derivative_of_softmax(x):
    s=softmax(x)
    return np.diag(s)-np.outer(s,s)

def plot_activations(activation,x):
    dict={'sigmoid':[sigmoid,derivative_of_sigmoid],
          'tanh':[tanh,derivative_of_tanh],
          'relu':[relu,derivative_of_relu],
          'leaky_relu':[leaky_relu,derivative_of_leaky_relu]}
    y1=dict[activation][0](x)
    y2=dict[activation][1](x)
    plt.figure(figsize=[8,8])
    plt.plot(x,y1,'r-',label=f"f(x)")
    plt.plot(x,y2,'b-',label=f"f'(x)")
    plt.title(f'{activation} Function')
    plt.xlabel('x')
    plt.ylabel("f(x)/f'(x)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():
    values = np.linspace(-10, 10, 100)
    plot_activations("sigmoid",values)
    plot_activations("tanh",values)
    plot_activations("relu",values)
    plot_activations("leaky_relu",values)

    sof=softmax(values)
    derv_sof=derivative_of_softmax(values)
    plt.figure(figsize=[8,8])
    plt.plot(values,sof,'r-',label=f"f(x)")
    plt.title(f'Softmax Function')
    plt.xlabel('x')
    plt.ylabel("f(x)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    print(f'Softmax values: {sof}')
    print(f'Derivative of Softmax values: {derv_sof}')

if __name__ == "__main__":
    main()