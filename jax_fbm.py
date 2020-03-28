# jax transposition
import jax.numpy as np
import matplotlib.pyplot as plt

from jax import vmap, jit

TAU = 2 * np.pi
SEED = 0
N_OCTAVES = 10
GAIN = .5
L = 2.

def mix(a, b, u):
    return (1 - u)*a + u*b

def cubic(x):
    return x*x*(3 - 2*x)

def quintic(x):
    return x*x*x*(10 + x*(-15 + 6*x))

# noise functions

def get_rand_fn(seed):
    a = np.array([12.9898, 78.233])
    def rand(p):
        r = np.sin(np.dot(seed + p, a)) * 43758.5453123
        return r - np.floor(r)
    return rand

noise_fn = get_rand_fn(SEED)

def value_noise(p, S, u):
    # p is point in 2d space, table is noise table, u is interpolation fn
    # p_frac, p_int = np.modf(p)
    p_int = np.floor(p)
    p_frac = p - p_int
    # p_int = (p_int % S).astype(np.int32)
    # print(p_int)
    a = noise_fn(p_int)
    b = noise_fn(p_int + np.array([1, 0]))
    c = noise_fn(p_int + np.array([0, 1]))
    d = noise_fn(p_int + np.array([1, 1]))
    # interpolate
    ux = u(p_frac[0])
    uy = u(p_frac[1])
    ab = mix(a, b, ux)
    cd = mix(c, d, ux)
    return mix(ab, cd, uy)

@jit
def fractional_brownian(p, octaves, gain, l):
    val = 0.
    g = 1.
    # print(p)
    q = p
    S = 2
    for i in range(10):
        # print('octave %s' % i)
        val += g*value_noise(q, S, cubic)
        q = q * l
        S *= l
        g *= gain
    # print(p)
    return val

a = 0
b = 1
xarr = np.arange(100) / 100

inter = lambda x, u: mix(a, b, u(x))

# plot interpolation function profiles
# yarr = inter(xarr, cubic)
# plt.plot(xarr, yarr)
# plt.show()
# yarr = inter(xarr, quintic)
# plt.plot(xarr, yarr)
# plt.show()

im_size = 3
grid_size = 1000
N = im_size * grid_size
A = np.arange(N, dtype=np.float32)
grid = np.stack((np.meshgrid(A, A)), -1) / grid_size
noisec = lambda p: value_noise(p, im_size, cubic)
noiseq = lambda p: value_noise(p, im_size, quintic)

p = np.array([0.1, 0.5])

print(noisec(p))
print(noiseq(p))

uvs = np.reshape(grid, (-1, 2))
nc_img = np.reshape(vmap(noisec)(uvs), (N, N))
plt.matshow(nc_img)
plt.show()

fbm = lambda p: fractional_brownian(p,
                                    N_OCTAVES,
                                    GAIN,
                                    L)

nc_img = np.reshape(vmap(fbm)(uvs), (N, N))
plt.matshow(nc_img)
plt.show()

compose = lambda p: fbm(p + fbm(p + fbm(p)))
compose = jit(compose)

ncc_img = np.reshape(vmap(compose)(uvs), (N, N))
plt.matshow(ncc_img)
plt.show()

# noisec_im = vmap(noisec)(grid)
# noiseq_im = vmap(noiseq)(grid)
# fig, axs = plt.subplots(2, 1)
# axs[0].imshow(noisec_im, interpolation='none')
# axs[1].imshow(noiseq_im, interpolation='none')
# plt.matshow(noise_im)
# plt.show()