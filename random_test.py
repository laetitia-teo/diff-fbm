import numpy as np
import matplotlib.pyplot as plt

def get_rand_fn(seed):
    a = np.array([12.9898, 78.233])
    def rand(p):
        r = np.sin(np.dot(seed + p, a)) * 43758.5453123
        return r - np.floor(r)
    return rand

A = np.arange(200)
x, y = np.meshgrid(A, A)
M = np.array([x, y]).swapaxes(0, -1)
f = get_rand_fn(0)
M = np.apply_along_axis(f, -1, M)
plt.matshow(M)
plt.show()