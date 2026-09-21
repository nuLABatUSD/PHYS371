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