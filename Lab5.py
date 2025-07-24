from Lab2 import Feed_forward_neural_network
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot(x, y,title):
    plt.figure(figsize=(6, 6))
    colours = ["red" if val <= 0.5 else "blue" for val in y]

    x1 = [i[0] for i in x]
    x2 = [i[1] for i in x]

    sns.scatterplot(x=x1, y=x2, hue=colours, palette={"red": "red", "blue": "blue"})
    plt.title("XOR Activation Output Colour-Coded")
    plt.xlabel("Input X1")
    plt.ylabel("Input X2")
    plt.legend(title="Output")
    plt.grid(True)
    plt.show()
X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=object)
fnn_results=[]
i=0
for x in X:
    fnn=Feed_forward_neural_network(input=x,number_of_neurons=[2,1],number_of_hidden_layers=2)
    fnn_results.append(fnn.run())
    i+=1
print(fnn_results)
plot(X,[0,1,1,0],"Original XOR Activation")
hidden_layer_coords = [layer['Layer 1'].flatten().astype(float) for layer in fnn_results]
final_outputs = [layer['Layer 2'].flatten().astype(float) for layer in fnn_results]
plot(hidden_layer_coords, final_outputs, "Projected XOR Activation (Layer 1)")