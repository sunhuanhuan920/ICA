"""
    Test bench for ICA class
    Arthor: Zhenhuan(Steven) Sun
"""

from ICA import ICA
import numpy as np
import scipy.io.wavfile

# load data
X = np.loadtxt("./Data/mix.dat")

# convert numerical matrix data back to sound wave file
for i in range(X.shape[1]):
    scipy.io.wavfile.write("./Signals/input/mixed_{}.wav".format(i), 11025, X[:, i])

# perform independet component analysis
ica = ICA(alpha=0.1, method="logistic", iterations=50)
ica.fit(X)
S = ica.transform(X)

# write recovered source back to wave file
for i in range(S.shape[0]):
    scipy.io.wavfile.write("./Signals/output/unmixed_{}.wav".format(i), 11025, S[i, :])