"""
    Independent Component Analysis Class
        - Laplace Sources
        - Logistic Sources
    Arthor: Zhenhuan(Steven) Sun
"""

import numpy as np
from scipy.special import expit # signmoid function

class ICA():
    def __init__(self, alpha=0.1, method="logistic", iterations=100):
        # learning rate set by user
        self.alpha = alpha

        # the assumption that user want to make to the souce signal
        # logistic source or laplace source
        self.method = method

        # the number of iterations to run gradient ascent algorithm
        self.epoch = iterations

    def fit(self, X):
        # number of examples(time points) and number of features(microphones that have recording)
        self.n_examples, self.n_features = X.shape

        # Initialize W for stochastic gradient ascent
        # normally I initialize this matrix to all zeros matrix using np.zeros
        # however if we do that in ICA, we would not be able to find the inverse of this matrix
        # since all zeros matrix is a singular matrix
        self.W = np.random.rand(self.n_features, self.n_features)

        # run stochastic gradient ascent for a certain number of epoch
        for e in range(self.epoch):
            print("Epoch: {}".format(e+1))
            # disrupt the order of training set
            random_order = np.random.permutation(self.n_examples)

            for i in random_order:
                # choose one example from the disrupted training set
                # and make it a column vector
                x = np.expand_dims(X[i, :], axis=1)

                if self.method == "logistic":
                    # gradient using logistic distribution (Assuming sources are logistic distributed)
                    gradient = np.linalg.inv(self.W.T) + (1 - 2 * expit(self.W.dot(x))).dot(x.T)

                elif self.method == "laplace":
                    # gradient using Laplace distribution (Assuming sources are laplace distributed)
                    gradient = np.linalg.inv(self.W.T) - np.sign(self.W.dot(x)).dot(x.T)

                # gradient ascent
                self.W = self.W + self.alpha * gradient
            
            print("\tCompleted")

    def transform(self, X):
        # recover the original sources
        S = self.W.dot(X.T)

        return S