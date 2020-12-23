import numpy as np

# y = 2 + 5x + epsilon(noise)
x = np.random.rand(100, 1)
y = 2 + 5 * x + .2 * np.random.randn(100, 1)
res = np.concatenate((x, y), axis=1)
np.savetxt("validation.csv", res, delimiter=",")