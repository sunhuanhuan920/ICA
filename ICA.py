import numpy as np
import scipy.io.wavfile

class ICA():
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

# load data
X = np.loadtxt("./Data/mix.dat")

# convert numerical matrix data back to sound wave file
for i in range(X.shape[1]):
    scipy.io.wavfile.write("Signals/input/mixed_{}.wav".format(i), 11025, X[:, i])