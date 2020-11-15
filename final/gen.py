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
    len_text = len(text)
    log_likelihood = np.zeros(0)
    for i in range(0, len_text-1):
        curr_c = text[i]
        next_c = text[i+1]
        prob = trans_matrix[CHAR_TO_IDX[curr_c]][CHAR_TO_IDX[next_c]]
        log_likelihood = np.append(log_likelihood, prob)
    
    # CHANGE LAER
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
    for index, row in df.iterrows():
        text = process_text(row["text"])
        matrix = gen_trans_matrix(text)
        matrices.append(matrix)
        
    df["matrix"] = matrices
    print(df.columns)
    print(df.loc[0, "matrix"])

process_df(sample_df)
    




