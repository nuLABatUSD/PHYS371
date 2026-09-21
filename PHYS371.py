import numpy as np
import matplotlib.pyplot as plt

def walk_mean(N_samples, N_walks):
  f = np.zeros(N_samples)
  for i in range(N_samples):
    dx = rg.choice([-1, 1], size=N_walks)
    t, x = random_walk(dx)
    f[i] = np.mean(x)
  return np.mean(f)