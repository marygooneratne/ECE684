import numpy as np

REPLACE = 'r'
INSERT = 'i'
DELETE = 'd'
TRANSPOSE = 't'

COMP = [
    ['id', 'di', 'rr'],
    ['dr', 'rd'],
    ['dd']
]


def compare(A, B, k=3):
    l1, l2 = len(A), len(B)

    if l1 < l2:
        l1, l2 = l2, l1
        A, B = B, A

    if l1 - l2 > 2:
        return 100

    models = COMP[l1-l2]

    for model in models:
        cost = check_model(A, B, l1, l2, model, k)

    if cost >= k:
        cost = 100

    return cost


def check_model(A, B, l1, l2, model, k):
    """Check if the model can transform A into B"""

    idx1, idx2 = 0, 0
    cost, pad = 0, 0
    while (idx1 < l1) and (idx2 < l2):
        if A[idx1] != B[idx2 - pad]:
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
                if (idx2 + 1) < l2 and A[idx1] == B[idx2+1]:
                    idx1 += 1
                    idx2 += 1
                    pad = 1
                else:
                    return k
        else:
            idx1 += 1
            idx2 += 1
            pad = 0

    return cost + (l1 - idx1) + (l2 - idx2)


if __name__ == "__main__":

    print(compare('aaaaaaaaaaapple', 'snapple'))
