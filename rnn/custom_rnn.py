class r_neural_network(object):
    def __init__(self, input_size=2, output_size=1, hidden_size=3, W1=None, W2=None):
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # weights
        if not W1.any() or not W2.any():
            print("NONE")
            # (3x2) weight matrix from input to hidden layer
            self.W1 = np.random.randn(self.input_size, self.hidden_size)
            # (3x1) weight matrix from hidden to output layer
            self.W2 = np.random.randn(self.hidden_size, self.output_size)
        else:
            self.W1 = W1
            self.W2 = W2

    def forward(self, X):
        # forward propagation through our network
        # dot product of X (input) and first set of 3x2 weights
        self.z = np.dot(X, self.W1)

        self.z2 = self.tanh(self.z)  # activation function

        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.W2)

        o = self.tanh(self.z3)  # final activation function
        return o

    def tanh(self, s):
        # activation function
        return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
