'''Use the first 10k tagged sentences from the Brown corpus to generate the
components of a part-of-speech hidden markov model: the transition matrix,
observation matrix, and initial state distribution. Use the universal tagset:
nltk.corpus.brown.tagged_sents(tagset=’universal’)[:10000]
Also hang on to the mappings between states/observations and indices. Include
an OOV observation and smoothing everywhere.'''
import nltk
import numpy as np

tag_set = set()
word_set = set()
tag_count = {}
obs_dict = {}
obs_matrix = []
trans_dict = {}
trans_matrix = []
word_map = {}
tag_map = {}

def gen_mappings():
    global tag_map
    global word_map

    word_map['START'] = 0 
    idx = 1
    for word in word_set:
        word_map[word] = idx
        idx += 1

    tag_map['START'] = 0 
    idx = 1
    for tag in tag_set:
        tag_map[tag] = idx
        idx += 1

def gen_counts(corpus):
    global trans_dict
    global word_set
    global tag_set
    global tag_count
    global obs_dict

    for sentence in corpus:
        previous = 'START'
        for couple in sentence:
            word = couple[0]
            tag = couple[1]
            tag_key = previous + ", " + tag
            obs_key = word + ", " + tag
            word_set.add(word)
            tag_set.add(tag)

            if tag in tag_count:
                tag_count[tag] += 1.0
            else:
                tag_count[tag] = 1.0
            
            if tag_key in trans_dict:
                trans_dict[tag_key] += 1.0
            else:
                trans_dict[tag_key] = 1.0
            
            if obs_key in obs_dict:
                obs_dict[obs_key] += 1.0
            else:
                obs_dict[obs_key] = 1.0
            
            previous = tag
    
    return trans_dict

def gen_transition_matrix():
    global trans_dict
    global trans_matrix
    trans_matrix = np.zeros((len(tag_set)+1,len(tag_set)+1))
    probs_dict = {}
    print(tag_count)
    for couple in trans_dict:
        key = couple.split(", ")[0]
        if key in probs_dict:
            probs_dict[key] += trans_dict[couple]
        else:
            probs_dict[key] = trans_dict[couple]
    
    for couple in trans_dict:
        tag1 = couple.split(", ")[0]
        tag2 = couple.split(", ")[1]
        trans_matrix[tag_map[tag1]][tag_map[tag2]] = trans_dict[couple]/probs_dict[key]
    
    for tag_i in tag_set:
        i = tag_map[tag_i]
        total = float(len(word_set) + tag_count[tag_i])
        for tag_j in tag_set:
            j = tag_map[tag_j]
            if trans_matrix[i][j] == 0.0:
                trans_matrix[i][j] = 1.0 / total
    
    return trans_matrix


def gen_observation_matrix():
    global tag_count
    global obs_dict
    global obs_matrix
    global tag_map
    global word_map
    
    obs_matrix = np.zeros((len(word_set)+1,len(tag_set)+1))

    for key in obs_dict:
        tag = key.split(", ")[1]
        word = key.split(", ")[0]
        obs_matrix[word_map[word]][tag_map[tag]] = obs_dict[key]/tag_count[tag]
    return obs_matrix

def gen_initial_distribution():
    return None

if __name__ == '__main__':
    corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    gen_counts(corpus)
    gen_mappings()
    print(gen_transition_matrix())
    print(gen_observation_matrix())