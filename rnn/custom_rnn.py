import numpy as np


class r_neural_network(object):
    def __init__(self, input_size=23, output_size=1, hidden_size=20, W1=None, W2=None):
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # weights
        if not W1 or not W2:
            print("NONE")
            # (23x20) weight matrix from input to hidden layer
            self.W1 = np.random.randn(self.input_size, self.hidden_size)
            # (20x3) weight matrix from hidden to output layer
            self.W2 = np.random.randn(self.hidden_size, self.output_size)
        else:
            self.W1 = W1
            self.W2 = W2

    def forward(self, X):
        X = np.concatenate((X, np.array(self.hidden_size * [0])), axis=None)
        # forward propagation through our network
        # dot product of X (input) and first set of 3x2 weights
        self.z = np.dot(X, self.W1)
        print(self.z)
        print(self.z.shape)
        # dot product of hidden layer (z2) and second set of 3x1 weights
        o = np.dot(self.z, self.W2)
        return o

    def softmax(self, s):
        # activation function
        return (np.exp(s) / np.sum(np.exp(s)))
