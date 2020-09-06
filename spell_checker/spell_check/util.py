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
from spell_check.comparator import compare
import numpy as np

DICT_DATA_2 = "./dict2.txt"
DICT_DATA = "./dict.txt"


def init_dict(DICT_FILE):
    f = open(DICT_FILE, "r")
    dict_set = set()
    for word in f:
        dict_set.add(word.split('\n')[0])
    return dict_set


def string_to_tokens(input_string):
    #     tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",input_string)
    tokens = input_string.split(" ")
    return tokens


def spell_check_str(input_string, dictionary):
    if len(input_string) == 0:
        return 1
    if re.match("[A-Z][a-z'-]+", str(input_string)):
        return 1
    if re.match("[\.\+\-\(\,\;\&\)\'\"\!\?]+", str(input_string)):
        return 1
    if input_string not in dictionary:
        return 0
    else:
        return 1


def calc_dist(A, B, k):
    return compare(A, B, k)


def calc_dist_deprecated(A, B, k=2):

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
            if(grid[x][y] >= k):
                return 100
    return grid[x][y]


def find_closest(input_string, dictionary, k):
    match_word = dictionary.pop()
    dist = calc_dist(input_string, match_word, k)
    for word in dictionary:
        if abs(len(word) - len(input_string)) >= dist:
            continue
        temp_dist = calc_dist(input_string, word, k)
        if temp_dist < dist:
            match_word = word
            dist = temp_dist
        if dist == 1:
            return match_word
    return match_word


def sub_word(word, dictionary, k):
    prefix = ''
    suffix = ''
    suffix2 = ''
    title = word.istitle()
    upper = word.isupper()
    mod_word = word
    if re.match("[\[\.\+\-\(\,\;\&\)\\'\\\"\!\?][\S\s]+", str(word)):
        prefix = word[0]
        mod_word = mod_word[1:]
    if re.match("[\s\S]+[\.\+\-\(\,\;\&\)\'\"\!\?\]]", str(word)):
        suffix2 = word[len(word)-1]
        mod_word = mod_word[0:len(mod_word)-1]
    mod_word = mod_word.split('\'')
    if len(mod_word) > 1:
        suffix = '\'' + mod_word[1]
    mod_word = mod_word[0]

    if mod_word.isdigit():
        return word
    mod_word = mod_word.lower()
    if not spell_check_str(mod_word, dictionary):
        closest = find_closest(mod_word, dictionary, k)
        if closest is not None:
            mod_closest = prefix + closest + suffix + suffix2
            if title:
                mod_closest = mod_closest.title()
            if upper:
                mod_closest = mod_closest.upper()
            return mod_closest
    return word


def correct_arr(arr, dictionary, k):
    for i, word in enumerate(arr):
        arr[i] = sub_word(word, dictionary, k)
    return " ".join(arr)


def spell_check_list(arr, dictionary):
    err = []
    for word in arr:
        if not spell_check_str(word, dictionary):
            err.append(word)
    return err


def autocorrect(word_dict, str, k=3):
    tokens = string_to_tokens(str)
    fixed = correct_arr(tokens, word_dict, k)
    return fixed


def run_test(test_file, k=10):
    word_dict = init_dict(DICT_DATA)
    file = open(test_file, 'r')
    fixed_lines = []
    for line in file:
        fixed_lines.append(autocorrect(word_dict, line.split('\n')[0], k))

    corrected = '\n'.join(fixed_lines)
    return corrected


def test():
    test_file = '../tests/austen-short.txt'
    print(run_test(
        test_file,
        3
    ))


if __name__ == "__main__":
    test()
