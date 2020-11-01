import random
import torch.nn as nn
import torch
import time
import math
import gen_gbu
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return hidden, output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def preprocess(X, Y):
    new_X = []
    new_Y = []
    good_tensor = [[1, 0, 0]]
    bad_tensor = [[0, 1, 0]]
    neutral_tensor = [[0, 0, 1]]
    for i, x in enumerate(X):
        new_x = []
        for word in x:
            if word == 'bad':
                new_x.append(bad_tensor)
            elif word == 'good':
                new_x.append(good_tensor)
            else:
                new_x.append(neutral_tensor)
        new_x = torch.tensor(new_x)
        new_X.append(new_x)
        y = Y[i]
        if y > 0:
            new_Y.append(torch.tensor([2], dtype=torch.long))
        if y == 0:
            new_Y.append(torch.tensor([1], dtype=torch.long))
        if y < 0:
            new_Y.append(torch.tensor([0], dtype=torch.long))
    return new_X, new_Y
