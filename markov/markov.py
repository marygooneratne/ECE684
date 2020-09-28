import re
import nltk
import numpy as np
import os

TRUMP_SPEECH = r"\data\speeches.txt"
OBAMA = r'\data\pres\obama\obama_speeches_'


def model_gen(n, tokens):

    n_grams = [[tokens[i] for i in range(j, j+n)]
               for j in range(0, len(tokens)-n)]
    _dict = {}
    # Build n-gram table
    for idx in range(0, len(n_grams)):
        key = tuple(n_grams[idx][0:len(n_grams[idx])-1])
        val = n_grams[idx][len(n_grams[idx])-1]
        if key not in _dict.keys():
            _dict[key] = {'tokens': [], 'P': []}
        if val not in _dict[key]['tokens']:
            _dict[key]['tokens'].append(val)
            _dict[key]['P'].append(0)

        _dict[key]['P'][_dict[key]['tokens'].index(val)] += 1

    # Normalize P
    for key in _dict.keys():
        _P = sum(_dict[key]['P'])
        _dict[key]['P'] = [p/_P for p in _dict[key]['P']]
    return _dict

# Stochastic selection


def non_det(loc):
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


def finish_sentence(sentence, n, corpus, deterministic=False):
    '''
        Args:
            sentence [list of tokens] that we’re trying to build on
            n [int], the length of n-grams to use for predictions
            corpus [list of tokens]
            deterministic [bool]: flag indicating whether the process should be deterministic
        Returns:
            an extended sentence until the first ., ?, or ! is found OR until it has 15 total tokens.
    '''
    # Init full gram model for grams 0 to n
    model = {i: model_gen(i, corpus) for i in range(2, n+1)}
    print(corpus)
    full = sentence
    print(model)

    # Iteratively append words
    while(full[len(full)-1] not in ['.', '?', '!'] and len(full) < 16):
        # Use last n-1 words for look-up
        key = full[len(full)-n+1:]
        key = tuple(key)

        # Search in first available dictionary
        N = n

        _dict = model[N]
        while key not in _dict.keys() and N > 2:
            N -= 1
            _dict = model[N]

        loc = _dict[key]
        next = det(loc) if deterministic else non_det(loc)
        full.append(next)

    return full


def finish_sentence(sentence, n, corpus, deterministic=False, length=15):
    '''
        Args:
            sentence [list of tokens] that we’re trying to build on
            n [int], the length of n-grams to use for predictions
            corpus [list of tokens]
            deterministic [bool]: flag indicating whether the process should be deterministic
        Returns:
            an extended sentence until the first ., ?, or ! is found OR until it has 15 total tokens.
    '''
    # Init full gram model for grams 0 to n
    model = {i: model_gen(i, corpus) for i in range(2, n+1)}
    full = sentence
    print(corpus)
    print(model)

    # Iteratively append words
    while(full[len(full)-1] not in ['.', '?', '!'] and len(full) < length):
        # Use last n-1 words for look-up
        key = full[len(full)-n+1:]
        key = tuple(key)

        # Search in first available dictionary
        N = n
        _dict = model[N]
        while key not in _dict.keys() and N > 2:
            N -= 1
            _dict = model[N]
        print(_dict)
        loc = _dict[key]
        next = det(loc) if deterministic else non_det(loc)
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


def get_speeches(file_prefix, len):
    words = []
    for i in range(0, len):

        path = os.getcwd()
        file_name = path + file_prefix + str(i).zfill(3) + '.txt'
        words = words + get_corpus(file_name)
    return words


if __name__ == "__main__":
    # sentence = ['she', 'was', 'not', 'in', 'this']
    # n = 3
    # corpus = [w.lower() for w in nltk.corpus.gutenberg.words(
    #     'austen-sense.txt')]
    # deterministic = True
    # print(finish_sentence(sentence, n, corpus, deterministic))

    # sentence = ['mexico', 'is']
    # n = 3
    # corpus = get_corpus(TRUMP_SPEECH)
    # deterministic = False
    # length = 200
    # print(" ".join(finish_sentence(sentence, n, corpus, deterministic, length)))

    sentence = ['i', 'am']
    n = 3
    corpus = get_speeches(OBAMA, 50)
    deterministic = False
    length = 200
    print(" ".join(finish_sentence(sentence, n, corpus, deterministic, length)))
