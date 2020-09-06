import numpy as np


def get_trial_result(sample):
    for i in range(len(sample)):
        if sample[i] > 0:
            return i


def lda(vocabulary, beta, alpha, xi):
    # for v in vocabulary:
    N = np.random.poisson(xi)

    theta = np.random.dirichlet(alpha, N)
    w = []
    for t in theta:
        topic = beta[get_trial_result(np.random.multinomial(1, t))]

        word = vocabulary[get_trial_result(np.random.multinomial(1, topic))]
        w.append(word)

    return w


vocabulary = ['bass', 'pike', 'deep', 'tuba', 'horn', 'catapult']
beta = np.array([
    [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
    [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
])
alpha = np.array([1, 3, 8])
xi = 50

print(lda(vocabulary, beta, alpha, xi))
