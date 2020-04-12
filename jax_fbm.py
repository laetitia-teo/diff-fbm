import time
import cv2
import numpy as onp
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
    p_int = np.floor(p)
    p_frac = p - p_int
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
    q = p
    S = 2
    for i in range(10):
        val += g*value_noise(q, S, cubic)
        q = q * l
        S *= l
        g *= gain
    return val

a = 0
b = 1
xarr = np.arange(100) / 100

inter = lambda x, u: mix(a, b, u(x))

im_size = 1
grid_size = 1000
N = int(im_size * grid_size)
A = np.linspace(0, im_size, N)
grid = np.stack((np.meshgrid(A, A)), -1)
noisec = lambda p: value_noise(p, im_size, cubic)
noiseq = lambda p: value_noise(p, im_size, quintic)

p = np.array([0.1, 0.5])

uvs = np.reshape(grid, (-1, 2))

fbm = lambda p: fractional_brownian(p,
                                    N_OCTAVES,
                                    GAIN,
                                    L)

# nc_img = np.reshape(vmap(fbm)(uvs), (N, N))
# plt.matshow(nc_img)
# plt.show()

compose = lambda p: fbm(p + fbm(p + fbm(p)))
compose = jit(compose)

@jit
def compose2d(p):
    q = p + np.array([
        fbm(p),
        fbm(p + np.array([5.2, 1.3]))])
    r = p + np.array([
        fbm(q + np.array([1.7, 9.2])),
        fbm(q + np.array([8.3, 2.8]))])
    return fbm(r)

@jit
def compose2dvar(p):
    q = p + np.array([
        fbm(p),
        fbm(p + np.array([5.2, 1.3]))])
    r = p + np.array([
        fbm(p + 4*q + np.array([1.7, 9.2])),
        fbm(p + 4*q + np.array([8.3, 2.8]))])
    return fbm(p + r)

@jit
def compose2dout(p):
    q = p + np.array([
        fbm(p),
        fbm(p + np.array([5.2, 1.3]))])
    r = p + np.array([
        fbm(p + 4*q + np.array([1.7, 9.2])),
        fbm(p + 4*q + np.array([8.3, 2.8]))])
    s = np.array([
        fbm(p + 4*q + r),
        fbm(p + 4*q + r + np.array([1.3, 5.2]))])
    return s

def get_fbm_compose(depth):
    def fbm_compose(p):
        pass

# ncc_img = np.reshape(vmap(compose)(uvs), (N, N))
# ncc_img2d = np.reshape(vmap(compose2dvar)(uvs), (N, N))
# fig, axs = plt.subplots(1, 2)
# axs[0].matshow(ncc_img)
# axs[1].matshow(ncc_img2d)
# plt.show()

# img2d = np.reshape(vmap(compose2dout)(uvs), (N, N, 2))
# ig, axs = plt.subplots(1, 2)
# axs[0].matshow(img2d[..., 0])
# axs[1].matshow(img2d[..., 1])
# plt.show()

# displacement experients

# weighing fn for managing border effects
w = lambda p: np.sin(np.pi * p) / np.pi

x = np.linspace(0, 1, 1000)
y = vmap(w)(x)
plt.plot(x, y)
plt.show()

# coord = grid + vmap(w)(grid) * img2d
# print(coord.shape)
# ig, axs = plt.subplots(1, 2)
# axs[0].matshow(coord[..., 0])
# axs[1].matshow(coord[..., 1])
# plt.show()

img = np.array(cv2.imread('images/lena.jpg'))
print(img.shape)
N, M = img.shape[0], img.shape[1]

grid = np.stack(np.meshgrid(np.arange(N),
                            np.arange(M)), -1)
xys = np.stack(np.meshgrid(np.linspace(0, 1, N),
                           np.linspace(0, 1, M)), -1)
flat = np.reshape(xys, (-1, 2))
perturb = vmap(compose2dout)(flat + np.array([3, 2]))
# perturb1 = vmap(compose2d)(flat)
perturb = np.reshape(perturb, (N, M, 2))
# perturb1 = np.reshape(perturb1, (N, M, 1))
# xys = xys + vmap(w)(xys)*perturb
xys = xys + 0.25 * perturb
xys = np.clip(xys, 0., 1. - 1e-9)
grid = (xys * N).astype(int)
pimg = img[grid[..., 0], grid[..., 1]]

cv2.imwrite('images/lena4.jpg', onp.array(pimg))

pimg = np.flip(pimg, -1)
pimg = np.transpose(pimg, (1, 0, 2))

plt.imshow(pimg)
plt.show()

# while True:
#     t0 = time.time()
#     a = vmap(compose)(uvs)
#     t = time.time() - t0
#     print(1 / t)
#     time.sleep(1)