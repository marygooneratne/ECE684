import numpy as np


def get_trial_result(sample):
    for i in range(len(sample)):
        if sample[i] > 0:
            return i


def lda(vocabulary, beta, alpha, xi):

    N = np.random.poisson(xi)

    theta = np.random.dirichlet(alpha, N)
    w = []
    for t in theta:
        topic = beta[get_trial_result(np.random.multinomial(1, t))]

        word = vocabulary[get_trial_result(np.random.multinomial(1, topic))]
        w.append(word)

    return w

