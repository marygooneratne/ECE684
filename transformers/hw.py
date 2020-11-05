import string

import random
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import math
import time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128


def process_data(count=-1):
    file = open('./data/words_short.txt', 'r')
    english_data = file.read().split("\n")[0:count]
    f2 = open('./data/pig_latin_short.txt', 'r')
    piglatin_data = f2.read().split("\n")[0:count]

    alphabet = list(string.ascii_lowercase)
    alphadict = {}
    for i, a in enumerate(alphabet):
        alphadict.update({a: i+1})

    english = []
    pig_latin = []
    max_length = 0
    max_length_pl = 0
    for i, word in enumerate(english_data):
        if len(word) > max_length:
            max_length = len(word)
        pl = piglatin_data[i]
        if len(pl) > max_length_pl:
            max_length_pl = len(pl)
    for i, word in enumerate(english_data):
        english.append(to_vector(word, alphadict, max_length))
        pig_latin.append(to_vector(pl, alphadict, max_length_pl))
    print('Processed')
    return english, pig_latin


def to_vector(word, alphadict, max_length):
    vec = []
    for letter in word:
        idx = alphadict[letter]
        vec.append(one_hot(idx))
    while len(vec) < max_length:
        vec.append(one_hot(0))
    return vec


def one_hot(index):
    vec = 27 * [0]
    vec[index] = 1
    return vec


def load_data(count=-1, train_split=0.7, val_split=0.2):
    random.seed(1)
    english, pig_latin = process_data(count)
    total = len(english)
    train_size = int(train_split * total)
    val_size = int((train_split+val_split) * total)
    c = list(zip(english, pig_latin))

    random.shuffle(c)

    english, pig_latin = zip(*c)

    X_train = english[0:train_size]
    Y_train = pig_latin[0:train_size]
    X_val = english[train_size:val_size]
    Y_val = pig_latin[train_size:val_size]
    X_test = english[val_size:]
    Y_test = pig_latin[val_size:]

    return torch.as_tensor(X_train), torch.as_tensor(X_test), torch.as_tensor(X_val), torch.as_tensor(Y_train), torch.as_tensor(Y_val), torch.as_tensor(Y_test)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hidden_dim,
                 num_layers,
                 dropout):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hidden_dim,
                           bidirectional=True, dropout=dropout)

    def forward(self, x):

        embedded = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 hidden_dim,
                 num_layers,
                 dropout
                 ):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,
                x,
                hidden,
                cell):

        x = x.unsqueeze(0)

        embedded = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(
            embedded, (hidden, cell))

        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                src,
                trg,
                teacher_forcing_ratio=0.5):

        batch_size = src.shape[1]
        max_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size,
                              trg_vocab_size).to(device)

        hidden, cell = self.encoder(src)

        x = trg[0]

        for t in range(1, max_len):

            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            guess = output.argmax(1)

            x = trg[t] if random.random(
            ) < teacher_forcing_ratio else guess

        return outputs
