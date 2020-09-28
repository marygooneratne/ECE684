import re
import nltk
import numpy as np
import os


def generate_model(n, tokens):

    n_grams = [[tokens[i] for i in range(j, j+n)]
               for j in range(0, len(tokens)-n)]
    dict = {}
    # Build n-gram table
    for idx in range(0, len(n_grams)):
        key = tuple(n_grams[idx][0:len(n_grams[idx])-1])
        val = n_grams[idx][len(n_grams[idx])-1]
        if key not in dict.keys():
            dict[key] = {'tokens': [], 'P': []}
        if val not in dict[key]['tokens']:
            dict[key]['tokens'].append(val)
            dict[key]['P'].append(0)

        dict[key]['P'][dict[key]['tokens'].index(val)] += 1

    # Normalize P
    for key in dict.keys():
        P = sum(dict[key]['P'])
        dict[key]['P'] = [p/P for p in dict[key]['P']]
    return dict

# Stochastic selection


def stoc(loc):
    P = loc['P']
    next_arr = loc['tokens']
    next = np.random.choice(next_arr, 1, P)[0]
    return next

# Deterministic selection


def det(loc):
    P = loc['P']
    next_arr = loc['tokens']
    max_P = max(P)
    next = ''

    # Locate max probabality, alphabetically ordered token
    for idx, _p in enumerate(P):
        if _p == max_P and (next == '' or next_arr[idx] < next):
            next = next_arr[idx]

    return next


def build_sentence(seed, n, corpus, deterministic=False):

    # Init full gram model for grams 0 to n
    model = {i: generate_model(i, corpus) for i in range(2, n+1)}

    full = seed

    # Iteratively append words
    while(full[len(full)-1] not in ['.', '?', '!'] and len(full) < 16):
        # Use last n-1 words for look-up
        key = full[len(full)-n+1:]
        key = tuple(key)

        num = n
        dict = model[num]
        while key not in dict.keys() and num > 2:
            num -= 1
            dict = model[num]

        loc = dict[key]
        next = det(loc) if deterministic else stoc(loc)
        full.append(next)

    return full


def get_corpus(file_name):
    words = []
    try:

        with open(file_name, 'r') as f:
            for line in f:
                for word in line.split():
                    words.append(word.lower())
        return words

    except:
        return words
