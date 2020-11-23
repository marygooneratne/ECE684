import numpy as np
import pandas as pd
import math

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
sample_data = pd.read_csv("data/sample.csv")
author_df = pd.DataFrame(train_data)["author"]
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
sample_df = pd.DataFrame(sample_data)

CHAR_ALLOW = [ " ", "-", "'", '"', ".", ","]
ALPHA_ALLOW = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
                       "q", "r", "s", "t", "u", "v", "w", "x", "y","z", "ö"]
CHARS = CHAR_ALLOW + ALPHA_ALLOW
CHARS.sort()
CHAR_TO_IDX = {}
IDX_TO_CHAR = {}

for index, char in enumerate(CHARS):
    CHAR_TO_IDX[char] = index
    IDX_TO_CHAR[index] = char

LEN_CHARS = len(CHARS)


def process_char(char):
    char = char.lower()
    if char.isnumeric():
        return " "
    if char.isalpha():
        if char in ALPHA_ALLOW:
            return char
        else: return "ö"
    if char in CHAR_ALLOW:
        return char
    else: return " "

def process_text(text):
    cleaned = ""
    for c in text:
        cleaned += process_char(c)
    return cleaned

def gen_trans_matrix(text):
    trans_matrix = np.zeros((LEN_CHARS, LEN_CHARS))
    for i in range(len(text)-1):
        curr_c = text[i]
        curr_id = CHAR_TO_IDX[curr_c]
        next_c = text[i+1]
        next_id = CHAR_TO_IDX[next_c]
        trans_matrix[curr_id][next_id] += 1
    
    row_sums = np.sum(trans_matrix, 1)
    
    for i in range(LEN_CHARS):
        row_sum = row_sums[i]
        if(row_sum == 0):
            row_sum = 1
        trans_matrix[i, :] = trans_matrix[i, :]/row_sum
    
    return trans_matrix

def log_likelihood(matrix, text):
    text = process_text(text)
    len_text = len(text)
    log_likelihood = np.zeros(0)
    for i in range(0, len_text-1):
        curr_c = text[i]
        next_c = text[i+1]
        prob = matrix[CHAR_TO_IDX[curr_c]][CHAR_TO_IDX[next_c]]
        log_likelihood = np.append(log_likelihood, prob)
    
    text = process_text(text)
    len_text = len(text) 
    log_likelihood = np.zeros(0)
    for i in range(0, len(text)-1):
        current_word = text[i]
        next_word = text[i+1]
        Step_probab = matrix[CHAR_TO_IDX[current_word] , CHAR_TO_IDX[next_word]]    
        log_likelihood = np.append(log_likelihood, Step_probab) 

    log_likelihood = np.log(log_likelihood)
    likelihood_neglect_special_case = 0
    inf_count = 0

    for i in range(len(log_likelihood)):
        if (log_likelihood[i]!= float("-inf")):
            likelihood_neglect_special_case = likelihood_neglect_special_case+log_likelihood[i] 
        else:
            inf_count = inf_count+1 

    log_likelihood_acc = np.where(log_likelihood == float("-inf"), 0,  log_likelihood)
    log_likelihood_acc = np.cumsum(log_likelihood_acc)
    return likelihood_neglect_special_case/(len_clean-inf_count), log_likelihood_acc

def process_df(df):
    matrices = []
    texts = {}
    for index, row in df.iterrows():
        author = row["author"]
        text = process_text(row["text"])
        if author not in texts.keys():
            texts[author] = ""
        texts[author] = texts[author] + text
    data = []
    for author in texts.keys():
        d = {"author": author, "text": texts[author], "transition_matrix": gen_trans_matrix(texts[author])}
        data.append(d)

    text = "This is what I saw in the glass: A thin, dark man of medium stature attired in the clerical garb of the Anglican church, apparently about thirty, and with rimless, steel bowed glasses glistening beneath a sallow, olive forehead of abnormal height."
    for d in data:
        print(d["author"], ": ", log_likelihood(d["transition_matrix"], text))
words = set()
for i, r in sample_df.iterrows():
    words.update(process_text(r["text"]).split(" "))
print(words)


process_df(sample_df)
    