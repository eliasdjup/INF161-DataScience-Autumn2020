import numpy as np
import matplotlib.pyplot as plt

# Binomial distribution parameters
n = 100
p = 0.3
num_samples = 100000

# Sample from a binomial distribution
samples = np.random.binomial(n, p, size=num_samples)
samples_rounded = np.round(samples)

# Plot a histogram of the samples
axis = np.arange(start=min(samples), stop=max(samples)+1)
plt.hist(samples, bins=axis)
plt.show()
