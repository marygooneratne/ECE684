# Build from scratch a spelling corrector in Python. It should include:
#   1. tokenization
#   2. edit distance-based non-word spelling correction
#   3. de-tokenization
# As an example use case, consider a version of Jane Austen’s Sense and
# Sensibility (available via nltk’s gutenberg corpus) corrupted by random insertions,
# deletions, and substitutions.
# Constraints:
#   • Your spelling correction function should accept a single string and return
#   a single string.
#   • Your spelling correction module may use only standard libraries and
#   numpy.
#   • To test it you may use the nltk.corpus package.
# You may work in a group of 1 or 2. Submissions will be graded without
# regard for the group size. Submit your solution in a .zip file including a
# Jupyter notebook (.ipynb file) demonstrating its usage.
import re
import numpy as np

DICT_DATA_2 = "dict2.txt"
DICT_DATA = "dict.txt"
EXAMPLE_STR = "The family of Dashwood had long been settled i Sussex \
        Their estete was large, and their residence was at Norlad Park,\
        in the centre of their property, where, for many generations,\
        they had lived in so respectable a manner as to engage \
        the general good opinion of their surrounding acquaintance.\
        The late owner of thfs estat was a single man, who lived\
        to a very advanced age, and who for many years of hijs life,\
        had a constant companion nd housekeeper in his sister.\
        But her death, which happened ten ryears beore his own,\
        produced a great alteration in his home; fuor gto supply\
        her lodss, he invited and eceivepd into his house the family\
        of his nephew Mr. Henry Dashwood, the legal inheritkr\
        of the Norland estate, and te lperqson to wsom he intended\
        to bequeath it.  In the society o his nephew and niece,\
        and theoir childrn, the old Gentaeman's das were\
        comfortably spent.  His attacsment to them all increased.\
        The consmant attention of Mr. and Mrs. Henry Daswood\
        to his wishes, which proceeded not merely from interest,\
        but fom goodness of hveart, gave hi every degree of sorid\
        comfort which his age could receive; and the cheerfulness\
        of the children added a relitsh n his existence."


def init_dict(FILE_NAME):
    f = open(FILE_NAME, "r")
    dict_set = set()
    for word in f:
        dict_set.add(word.split('\n')[0])
    return dict_set

def string_to_tokens(input):
    tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",input)
    return tokens

def spell_check(input, dictionary):
    if input not in dictionary:
        return 0
    else: return 1

def calc_dist(A, B):
    # Initialize matrix
    xs = len(A)+1
    ys = len(B)+1
    grid = np.zeros((xs, ys), dtype=int)

    for i in range(1, xs):
        for j in range(1, ys):
            grid[i][0] = i
            grid[0][j] = j

    # Find cost of deletions,insertions and/or substitutions
    for x in range(1, xs):
        for y in range(1, ys):
            if A[x-1] == B[y-1]:  # Letters are the same
                cost = 0
            else:
                cost = 1
            grid[x][y] = min(grid[x-1][y] + 1,      # Cost of deletions
                             grid[x][y-1] + 1,          # Cost of insertions
                             grid[x-1][y-1] + cost)     # Cost of substitutions
    return grid[x][y]

def spell_check_list(arr, dictionary):
    err = []
    for word in arr:
        if not spell_check(word, dictionary):
            err.append(word)
    return err

def find_closest(input, dictionary):
    match_word = dictionary.pop()
    dist = calc_dist(input, match_word)
    for word in dictionary:
        temp_dist = calc_dist(input, word)
        if temp_dist < dist:
            match_word = word
            dist = temp_dist
        if dist == 1:
            return (match_word, dist)
    return (match_word, dist)

def sub_err(arr, dictionary):
    subs = []
    for word in arr:
        if not spell_check(word, dictionary):
            subs.append(find_closest(word, dictionary))
    return subs
        
        
if __name__ == "__main__":
    word_dict = init_dict(DICT_DATA)
    tokens = string_to_tokens(EXAMPLE_STR)
    err = spell_check_list(tokens, word_dict)
    subs = sub_err(tokens, word_dict)
    print(subs)
    # print(err)
    # print(word_dict)
    # print(tokens)
    # print(word_dict)
