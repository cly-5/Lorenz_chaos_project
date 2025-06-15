import numpy as np
import matplotlib.pyplot as plt

sigma = 10.0
rho = 28.0
beta = 8/3


def lorenz(state, t):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def rk4_step(f, state, t, dt):
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


dt = 0.01
t = np.arange(0, 30, dt)

init1 = np.array([1.0, 1.0, 1.0])
init2 = np.array([1.0 + 1e-6, 1.0, 1.0])

traj1 = np.zeros((len(t), 3))
traj2 = np.zeros((len(t), 3))
traj1[0] = init1
traj2[0] = init2

for i in range(len(t) - 1):
    traj1[i+1] = rk4_step(lorenz, traj1[i], t[i], dt)
    traj2[i+1] = rk4_step(lorenz, traj2[i], t[i], dt)

plt.figure()
plt.plot(t, traj1[:, 0], label='(1.0, 1.0, 1.0)')
plt.plot(t, traj2[:, 0], label='(1.0 + 1e-6, 1.0, 1.0)')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Lorenz Trajectories: x(t) for Two Close Initial Conditions')
plt.legend()
plt.grid(True)
plt.show()

diff = np.linalg.norm(traj1 - traj2, axis=1)
plt.figure()
plt.plot(t, diff)
plt.xlabel('Time')
plt.ylabel('||Δ(state)||')
plt.title('Divergence of Trajectories Over Time')
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.show()
