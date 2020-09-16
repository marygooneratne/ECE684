import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel as ldamodel
from gensim.models.ldamodel import LdaState as ldastate
def lda(vocabulary, beta, alpha, xi):
    '''
    Args:
        vocabulary - list (of length V ) of strings
        beta - topic-word matrix, numpy array of size (k, V )
        alpha - topic distribution parameter vector, of length k
        xi - Poisson parameter (scalar) for document size distribution
    Returns:
        w - list of words (strings) in a document
    '''
    n = np.random.poisson(xi)
    theta = np.random.dirichlet(alpha, xi)
    doc = []
    for i in theta:
        t = beta[int(np.nonzero(np.random.multinomial(1, i))[0])]
        w = vocabulary[int(np.nonzero(np.random.multinomial(1, t))[0])]
        doc.append(w)
    return doc

def corpus(vocabulary, beta, alpha, xi, num_docs):
    corpus = []
    for i in range(0, num_docs):
        corpus.append(lda(vocabulary, beta, alpha, xi))
    return corpus

def train_LDA(texts):
    common_dictionary = Dictionary(texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in texts]
    lda = ldamodel(common_corpus, alpha='auto', eta='auto')
    alpha= lda.alpha
    eta = lda.eta
    theta = np.random.dirichlet(lda.alpha, 50)
    beta = np.random.dirichlet(lda.eta, 3)
    print("ALPHA: ", alpha)
    print("THETA: ", theta)
    print("ETA: ", eta)
    print("BETA: ", beta)

if __name__ == "__main__":
    vocabulary = ['bass', 'pike', 'deep', 'tuba', 'horn', 'catapult']
    beta = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = np.array([1, 3, 8])
    xi = 50
    num_docs = 10000
    gen = lda(vocabulary, beta, alpha, xi)
    corpus = corpus(vocabulary, beta, alpha, xi, num_docs)
    train_LDA(corpus)

