"""
    Test bench for ICA class
    Signal visualization of both mixed and unmixed signals
    Arthor: Zhenhuan(Steven) Sun
"""

from ica import ICA
import numpy as np
import scipy.io.wavfile # for reading and writing wave file
import matplotlib.pyplot as plt

# load data
X = np.loadtxt("./Data/mix.dat")

# lower down the signal voice
X = X / np.max(np.abs(X))

# convert numerical matrix data back to sound wave file
for i in range(X.shape[1]):
    scipy.io.wavfile.write("./Signals/input/mixed_signal_{}.wav".format(i), 11025, X[:, i])

# perform independet component analysis on X
ica = ICA(alpha=0.1, method="logistic", iterations=10)
ica.fit(X)
S = ica.transform(X)

# write recovered source back to wave file
for i in range(S.shape[0]):
    scipy.io.wavfile.write("./Signals/output/unmixed_signal_{}.wav".format(i), 11025, S[i, :])

# number of samples in audio signal
n_samples = X.shape[0]
# sampling rate of audio signal (11025 samples/s)
sample_rate = 11025
# duration of audio signal in second
duration = n_samples / sample_rate
# extend the duration to evenly spaced values to represent time
time = np.arange(0, duration, 1/sample_rate)

# plot audio signal
fig, axes = plt.subplots(5, 2)
for r in range(5):
    for c in range(2):
        if c == 0:
            axes[r, c].plot(time, X[:, r])
            axes[r, c].set_title("Mixed Signal {}".format(r))
        else:
            axes[r, c].plot(time, S[r, :])
            axes[r, c].set_title("Unmixed Signal {}".format(r))

for ax in axes.flat:
    ax.set(xlabel="Time (s)", ylabel="Amplitude")

plt.show()