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
import numpy as np

DICT_DATA = "dict2.txt"


def init_dict(FILE_NAME):
    f = open(FILE_NAME, "r")
    dict_set = set()
    for word in f:
        dict_set.add(word.split('\n')[0])
    return dict_set


def calc_dist(A, B):

    # Initialize matrix
    xs = len(A)+1
    ys = len(B)+1
    grid = np.zeros((xs, ys), dtype=int)

    # Make the yumns
    for i in range(1, xs):
        for j in range(1, ys):
            grid[i][0] = i
            grid[0][j] = j

    # Find cost of deletions,insertions and/or substitutions
    for x in range(1, xs):
        for y in range(1, ys):
            if A[x-1] == B[y-1]:  # Letters on the same
                cost = 0
            else:
                cost = 1
            grid[x][y] = min(grid[x-1][y] + 1,      # Cost of deletions
                             grid[x][y-1] + 1,          # Cost of insertions
                             grid[x-1][y-1] + cost)     # Cost of substitutions

    print(grid[x][y])


if __name__ == "__main__":
    word_dict = init_dict(DICT_DATA)
    # print(word_dict)
    A = 'Apple Inc.'
    B = 'apple Inc'

    calc_dist(A.lower(), B.lower())
