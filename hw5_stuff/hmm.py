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
corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]

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

def viterbi(pi, a, b):
    global corpus
    obs = []
    obs.append(0)
    for word in corpus[0]:
        obs.append(word_map[word[0]])
    
    print(obs)
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

# def viterbi(obs, pi, A, B):
#     '''
#         Implement a function viterbi() that takes arguments:
#         1. obs - the observations [list of ints]
#         2. pi - the initial state probabilities [list of floats]
#         3. A - the state transition probability matrix [2D numpy array]
#         4. B - the observation probability matrix [2D numpy array]
#         and returns:
#         1. states - the inferred state sequence [list of ints]
#     '''


if __name__ == '__main__':
    corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    gen_counts(corpus)
    gen_mappings()
    trans = gen_transition_matrix()
    obs = gen_observation_matrix()
    print(viterbi(trans[0], trans, obs))
