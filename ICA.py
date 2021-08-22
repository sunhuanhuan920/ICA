import numpy as np
import scipy.io.wavfile

# load data
X = np.loadtxt("./Data/mix.dat")
print(X.shape)