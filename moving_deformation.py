import time
import cv2
import moviepy.editor as mpy

import numpy as onp
import jax.numpy as np

from jax import vmap, jit

TAU = 2 * np.pi
SEED = 0
N_OCTAVES = 10
GAIN = .5
L = 2.
T = 200. # period of the oscillations

def mix(a, b, u):
    return (1 - u)*a + u*b

def cubic(x):
    return x*x*(3 - 2*x)

def quintic(x):
    return x*x*x*(10 + x*(-15 + 6*x))

# noise functions

def noise_fn(p, seed):
    a = np.array([12.9898, 78.233])
    r = np.sin(np.dot(seed + p, a)) * 43758.5453123
    return r - np.floor(r)

# noise_fn = get_rand_fn(SEED)

def value_noise(p, seed, S, u):
    # p is point in 2d space, table is noise table, u is interpolation fn
    p_int = np.floor(p)
    p_frac = p - p_int
    a = noise_fn(p_int, seed)
    b = noise_fn(p_int + np.array([1, 0]), seed)
    c = noise_fn(p_int + np.array([0, 1]), seed)
    d = noise_fn(p_int + np.array([1, 1]), seed)
    # interpolate
    ux = u(p_frac[0])
    uy = u(p_frac[1])
    ab = mix(a, b, ux)
    cd = mix(c, d, ux)
    return mix(ab, cd, uy)

def fbm(p, seed):
    val = 0.
    g = 1.
    q = p
    S = 2
    for i in range(10):
        val += g * value_noise(q, seed, S, cubic)
        q = q * L
        S *= L
        g *= GAIN
    return val

@jit
def compose2dout(p, seed):
    q = p + np.array([
        fbm(p, seed),
        fbm(p + np.array([5.2, 1.3]), seed)])
    r = p + np.array([
        fbm(p + 4*q + np.array([1.7, 9.2]), seed),
        fbm(p + 4*q + np.array([8.3, 2.8]), seed)])
    s = np.array([
        fbm(p + 4*q + r, seed),
        fbm(p + 4*q + r + np.array([1.3, 5.2]), seed)])
    return s

# additional params

shift = np.array([3, 2])
img = np.array(cv2.imread('images/lena.jpg'))
img = np.transpose(img, (1, 0, 2))
N, M = img.shape[0], img.shape[1]
grid = np.stack(np.meshgrid(np.arange(N),
                            np.arange(M)), -1)

@jit
def render_one(seed):
    xys = np.stack(np.meshgrid(np.linspace(0, 1, N),
                               np.linspace(0, 1, M)), -1)
    flat = np.reshape(xys, (-1, 2))
    seed = np.ones_like(flat) * seed
    perturb = vmap(compose2dout)(flat + shift, seed)
    perturb = np.reshape(perturb, (N, M, 2))
    xys = xys + 0.25 * perturb
    xys = np.clip(xys, 0., 1. - 1e-9)
    grid = (xys * N).astype(int)
    pimg = img[grid[..., 0], grid[..., 1]]
    return pimg

def render_frame(t):
    # wrapper for original numpy conversion
    seed = 0.01 * np.sin(TAU * t / T)
    return onp.asarray(render_one(seed))

# for i in range(100):
#     t0 = time.time()
#     render_one(i)
#     t = time.time() - t0
#     print(f'time elapsed {t}')
#     print(f'fps {1/t}')

# initialize stuff

clip = mpy.VideoClip(render_frame, duration=10)
clip.write_videofile('output.mp4', fps=20)