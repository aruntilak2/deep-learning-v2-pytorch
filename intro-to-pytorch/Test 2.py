import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    return 1 if t>=0 else 0

def prediction(x_k, W, b):
    # x_k is the k-th point (x1, x2) in the data X, i.e. x_k = X[k]
    return stepFunction(np.matmul(x_k, W)+b)

# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate=0.01):
    for x_k, y_k in zip(X, y):
        dy_k = y_k - prediction(x_k, W, b)
        W = W + learn_rate * dy_k * x_k[:, None]
        b = b + learn_rate * dy_k
    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_max = X[:,0].max()
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = np.zeros((num_epochs, 2))
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines[i] = [-W[0]/W[1], -b/W[1]]  # [slope, intercept]
    return boundary_lines


if __name__ == '__main__':
    
    # Import data
    df = pd.read_csv('data.csv', sep=',', header=None, names=['x1', 'x2', 'y'])
    X = df[['x1', 'x2']].values
    y = df['y'].values
    
    # Boundary lines for each epoch
    boundary_lines = trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25)
    
    # Plotting
    x = np.linspace(-0.5, 1.5)
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    
    # plot points
    ax.scatter(df.loc[:49,'x1'], df.loc[:49,'x2'], label='class 0', marker='o')
    ax.scatter(df.loc[50:,'x1'], df.loc[50:,'x2'], label='class 1', marker='o')
    
    # plot boundary lines
    for (slope, intercept) in boundary_lines[:-1]:
        ax.plot(x, slope*x + intercept, ls='--', color='C2')
    ax.plot(x, boundary_lines[-1][0]*x + boundary_lines[-1][1], color='k')
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend(loc='best')
    plt.show()