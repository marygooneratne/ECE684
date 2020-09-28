import nltk
import sys
import math
from decimal import *
import numpy as np
tagset = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]

tag_list = set()
word_set = set()
transition_dict = {}
tag_count = {}
for value in tagset:
    previous = "start"
    for data in value:
        word_set.add(data[0].lower())
        tag = data[1]
        tag_list.add(tag)

        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1

        if (previous, tag) in transition_dict:
            transition_dict[(previous, tag)] += 1
            previous = tag
        else:
            transition_dict[(previous, tag)] = 1
            previous = tag

word_map = {'start': 0}
i = 1
for word in word_set:
    if word in word_map:
        continue
    else:
        word_map[word] = i
        i += 1
word_map['undefined'] = len(word_map)

i = 1
tag_map = {'start': 0}
for tag in tag_list:
    if tag in tag_map:
        continue
    else:
        tag_map[tag] = i
        i += 1

count_dict = transition_dict
prob_dict = {}
for key in count_dict:
    den = 0
    val = key[0]
    for key_2 in count_dict:
        if key_2[0] == val:
            den += count_dict[key_2]
    prob_dict[key] = Decimal(count_dict[key])/(den)

transition_prob = prob_dict
for tag in tag_list:
    if ("start", tag) not in transition_prob:
        transition_prob[("start", tag)] = Decimal(
            1) / Decimal(len(word_set) + tag_count[tag])
for tag1 in tag_list:
    for tag2 in tag_list:
        if (tag1, tag2) not in transition_prob:
            transition_prob[(tag1, tag2)] = Decimal(
                1) / Decimal(len(word_set) + tag_count[tag1])

train_data = tagset
count_word = {}
for value in train_data:
    for data in value:
        word = data[0]
        tag = data[1]
        if (word, tag) in count_word:
            count_word[(word, tag)] += 1
        else:
            count_word[(word, tag)] = 1

word_count = count_word
emission_prob_dict = {}
for key in word_count:
    emission_prob_dict[key] = Decimal(
        word_count[key])/tag_count[key[1]]

transition_matrix = np.zeros((len(tag_map), len(tag_map)))
# print(transition_prob)
for key, value in transition_prob.items():
    i = tag_map[key[0]]
    j = tag_map[key[1]]
    transition_matrix[i, j] = value
# print(transition_matrix)

observation_matrix = np.zeros((len(tag_map), len(word_map)+1))

for key, value in emission_prob_dict.items():
    i = tag_map[key[1]]
    j = word_map[key[0].lower()]

    observation_matrix[i][j] = value
random_model = 1/len(tag_map)
for tag in tag_map.keys():
    i = tag_map[tag]
    j = word_map['undefined']

    observation_matrix[i][j] = random_model


def viterbi(obs, pi, a, b):
    print(obs)
    # obs.append(0)
    # obs_plain = []
    # for word in tagset[0]:
    #     obs_plain.append(word)
    #     obs.append(word_map[word[0].lower()])
    # print(obs_plain)
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = np.zeros(T, dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])

    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]

    return path


pi = transition_matrix[0]
a = transition_matrix
b = observation_matrix

# states = viterbi([], pi, a, b)[0]
# final = []
# for state in states:
#     for tag in tag_map.keys():
#         if tag_map[tag] == state:
#             final.append(tag)
# print(final[1:])


def build_sentence(sentence):
    idx_sentence = [0]
    for wordtag in sentence:
        word = wordtag[0]
        try:
            idx = word_map[word]
        except:
            idx = word_map['undefined']
        idx_sentence.append(idx)
    return idx_sentence


def dec_sentence(states):
    final = []
    for state in states:
        for tag in tag_map.keys():
            if tag_map[tag] == state:
                final.append(tag)
    return final[1:]


test = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
print(test)
for t in test:
    print(t)
    idx_sentence = build_sentence(t)
    vt = viterbi(idx_sentence, pi, a, b)
    print(dec_sentence(vt))
    print()
