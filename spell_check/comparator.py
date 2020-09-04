import numpy as np

REPLACE = 'r'
INSERT = 'i'
DELETE = 'd'
TRANSPOSE = 't'

MATRIX = [
    ['id', 'di', 'rr'],
    ['dr', 'rd'],
    ['dd']
]

MATRIX_T = [
    ['id', 'di', 'rr', 'tt', 'tr', 'rt'],
    ['dr', 'rd', 'dt', 'td'],
    ['dd']
]


def compare(str1, str2, k=3, transpose=False):
    len1, len2 = len(str1), len(str2)

    if len1 < len2:
        len1, len2 = len2, len1
        str1, str2 = str2, str1

    if len1 - len2 > 2:
        return -1

    if transpose:
        models = MATRIX_T[len1-len2]
    else:
        models = MATRIX[len1-len2]

    for model in models:
        cost = check_model(str1, str2, len1, len2, model, k)

    if cost > k:
        cost = -1

    return cost


def check_model(str1, str2, len1, len2, model, k):
    """Check if the model can transform str1 into str2"""

    idx1, idx2 = 0, 0
    cost, pad = 0, 0
    while (idx1 < len1) and (idx2 < len2):
        if str1[idx1] != str2[idx2 - pad]:
            cost += 1
            if 2 < cost:
                return cost

            option = model[cost-1]
            if option == DELETE:
                idx1 += 1
            elif option == INSERT:
                idx2 += 1
            elif option == REPLACE:
                idx1 += 1
                idx2 += 1
                pad = 0
            elif option == TRANSPOSE:
                if (idx2 + 1) < len2 and str1[idx1] == str2[idx2+1]:
                    idx1 += 1
                    idx2 += 1
                    pad = 1
                else:
                    return k
        else:
            idx1 += 1
            idx2 += 1
            pad = 0

    return cost + (len1 - idx1) + (len2 - idx2)


if __name__ == "__main__":

    print(compare('aaaaaaaaaaapple', 'snapple'))
