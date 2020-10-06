"""Generate XOR data."""
import matplotlib.pyplot as plt
import numpy as np


def gen_xor():
    """Generate XOR data."""
    nobs_per = 50

    X = np.vstack((
        np.random.randn(nobs_per, 2) * 0.25 + [[0, 0]],
        np.random.randn(nobs_per, 2) * 0.25 + [[1, 1]],
        np.random.randn(nobs_per, 2) * 0.25 + [[0, 1]],
        np.random.randn(nobs_per, 2) * 0.25 + [[1, 0]],
    ))
    Y = np.hstack((
        np.zeros((2 * nobs_per,)),
        np.ones((2 * nobs_per,)),
    ))
    return X, Y


def main():
    """Plot XOR data."""
    X, Y = gen_xor()
    for y in np.unique(Y):
        plt.plot(X[Y == y, 0], X[Y == y, 1], 'o')
    plt.show()


if __name__ == "__main__":
    main()
