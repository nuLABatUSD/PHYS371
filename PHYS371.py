import numpy as np
import matplotlib.pyplot as plt

def random_walk(steps):
  if np.isscalar(steps):
    N = 1
  else:
    N = len(steps)
  x = np.zeros(N+1, dtype=float)
  x[1:] = np.cumsum(steps)
  t = np.arange(len(x), dtype=float)
  return t, x

def chi_square_acceleration(t, x, sigma, v0, a):
  return np.sum((x - (v0*t + 0.5*a*t**2))**2 / sigma**2)

def MC_step(rng, t, x, dx, v, a, prev_chi, dv, da):
  v_temp = rng.normal(loc=v, scale=dv)
  a_temp = rng.normal(loc=a, scale=da)

  chi_temp = chi_square_acceleration(t, x, dx, v_temp, a_temp)

  if chi_temp < prev_chi:
    return v_temp, a_temp, chi_temp
  else:
    if rng.random() < np.exp(-(chi_temp - prev_chi)/2):
      return v_temp, a_temp, chi_temp
    else:
      return v, a, prev_chi
