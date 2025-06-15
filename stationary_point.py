import numpy as np
import matplotlib.pyplot as plt


sigma = 10.0
rho = 28.0
beta = 8/3


def lorenz(state, t):
    x, y, z = state
    return np.array([
        sigma*(y - x),
        x*(rho - z) - y,
        x*y - beta*z
    ])


def rk4_step(f, state, t, dt):
    k1 = f(state, t)
    k2 = f(state + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(state + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(state + dt*k3, t + dt)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


dt = 0.01
t = np.arange(0, 30, dt)

initial_conditions = [
    (np.sqrt(72),             np.sqrt(72),             27.0),
    (np.sqrt(72) + 1e-6,      np.sqrt(72) + 1e-6,      27.0 + 1e-6),
    (1e-6,                    1e-6,                    1e-6)
]

display_labels = [
    r'$(\sqrt{72},\ \sqrt{72},\ 27)$',
    r'$(\sqrt{72}+10^{-6},\ \sqrt{72}+10^{-6},\ 27+10^{-6})$',
    r'$(10^{-6},\ 10^{-6},\ 10^{-6})$'
]

fig = plt.figure(figsize=(15, 5))
for idx, init in enumerate(initial_conditions):
    traj = np.zeros((len(t), 3))
    traj[0] = init
    for i in range(len(t)-1):
        traj[i+1] = rk4_step(lorenz, traj[i], t[i], dt)
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
    ax.set_title(f'Initial Condition {display_labels[idx]}', pad=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
