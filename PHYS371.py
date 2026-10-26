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

def MCMC_step(rng, t, x, dx, v, a, prev_chi, dv, da):
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

g_pendulum = 9.8
L_pendulum = 0.1
w02_pendulum = g_pendulum/L_pendulum
f0_pendulum = np.sqrt(w02_pendulum) / (2 * np.pi)
dt_pendulum = 0.01 / f0_pendulum / 2

@nb.njit()
def f(x, F, gamma = 0.1):
    der = np.zeros(2)
    der[0] = x[1]
    der[1] = - w02_pendulum * np.sin(x[0]) + F - gamma * x[1]
    return der

@nb.njit()
def RK4(x, F3, gamma = 0.1):
    k1 = f(x, F3[0], gamma)
    k2 = f(x + 0.5 * dt_pendulum * k1, F3[1], gamma)
    k3 = f(x + 0.5 * dt_pendulum * k2, F3[1], gamma)
    k4 = f(x + dt_pendulum * k3, F3[2], gamma)

    return x + dt_pendulum * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

@nb.njit()
def time_array(N_steps):
    N = np.arange(N_steps)
    return dt_pendulum * N

@nb.njit()
def force_time_array(N_steps):
    N = np.arange(2 * N_steps - 1)
    return dt_pendulum * N / 2

@nb.njit()
def ForcedPendulum(initial_displacement, N_steps, Force, gamma=0.1):
    if len(Force) != 2 * N_steps - 1:
        print("length of the Force array must be 2 N - 1")
        return None

    x = np.zeros((2, N_steps))
    x[0,0] = initial_displacement

    for i in range(N_steps-1):
        x[:,i+1] = RK4(x[:,i], Force[2*i:2*i+3], gamma)

    return time_array(N_steps), x[0,:]

@nb.njit()
def FreePendulum(initial_displacement, N_steps):
    F = np.zeros_like(force_time_array(N_steps))
    return ForcedPendulum(initial_displacement, N_steps, F, 0)