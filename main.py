"""
differentiable fbms - numpy version
"""
import numpy as np
import matplotlib.pyplot as plt

TAU = 2 * np.pi
SEED = 1
N_OCTAVES = 10
GAIN = .5
L = 2.

np.random.seed(SEED)

def mix(a, b, u):
    return (1 - u)*a + u*b

def cubic(x):
    return x*x*(3 - 2*x)

def quintic(x):
    return x*x*x*(10 + x*(-15 + 6*x))

# noise functions

# change this to get more controllable pseudorandom, and faster, and with no
# memory
def get_random_table(S):
    # shader implementations use sines with irrational ratios, because it is 
    # faster ?
    return np.random.random((S + 1, S + 1))
    
def value_noise(p, table, S, u):
    # p is point in 2d space, table is noise table, u is interpolation fn
    p_frac, p_int = np.modf(p)
    p_int = (p_int % S).astype(int)
    # print(p_int)
    # get table values
    a = table[tuple(p_int)]
    b = table[tuple(p_int + np.array([1, 0]))]
    c = table[tuple(p_int + np.array([0, 1]))]
    d = table[tuple(p_int + np.array([1, 1]))]
    # interpolate
    ux = u(p_frac[0])
    uy = u(p_frac[1])
    ab = mix(a, b, ux)
    cd = mix(c, d, ux)
    return mix(ab, cd, uy)

def fractional_brownian(p, octaves, gain, l, noise_fn, table_list):
    val = 0.
    g = 1.
    # print(p)
    q = p
    S = 2
    for i in range(octaves):
        # print('octave %s' % i)
        val += g*noise_fn(q, table_list[i], S, cubic)
        q = q * l
        S *= l
        g *= gain
    # print(p)
    return val

# tests

a = 0
b = 1
xarr = np.arange(100) / 100

inter = lambda x, u: mix(a, b, u(x))

# yarr = inter(xarr, cubic)
# plt.plot(xarr, yarr)
# plt.show()
# yarr = inter(xarr, quintic)
# plt.plot(xarr, yarr)
# plt.show()

im_size = 2
grid_size = 500
N = im_size * grid_size
A = np.arange(N)
grid = np.stack((np.meshgrid(A, A)), -1) / grid_size
t = get_random_table(im_size)
# noisec = lambda p: value_noise(p, t, im_size, cubic)
# noiseq = lambda p: value_noise(p, t, im_size, quintic)
# noisec_im = np.apply_along_axis(noisec, -1, grid)
# noiseq_im = np.apply_along_axis(noiseq, -1, grid)
# fig, axs = plt.subplots(2, 1)
# axs[0].imshow(noisec_im, interpolation='none')
# axs[1].imshow(noiseq_im, interpolation='none')
# # plt.matshow(noise_im)
# plt.show()

# fbm

p = np.array([0.1, 0.5])

table_list = []
S = im_size
for i in range(N_OCTAVES):
    table_list.append(get_random_table(S))
    S = int(S * L)

fbm = lambda p: fractional_brownian(p,
                                    N_OCTAVES,
                                    GAIN,
                                    L,
                                    value_noise,
                                    table_list)
# fbm_im = np.apply_along_axis(fbm, -1, grid)
# plt.matshow(fbm_im)
# plt.show()

grid = np.stack((np.meshgrid(A, A)), -1) / grid_size
compose = lambda p: fbm(p + fbm(p + fbm(p)))
c_im = np.apply_along_axis(compose, -1, grid)
print(compose(p))
plt.matshow(c_im)
plt.show()