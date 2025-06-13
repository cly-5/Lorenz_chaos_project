import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Simulation and plot the Lorenz energy

# Define lorenz equations
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma*(y-x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return np.array([dxdt, dydt, dzdt])


# Set initial condition, the parameters, and time steps
init_state = [1.0, 1.0, 1.0]

t_span = (0, 500)
t0, tn = t_span
N = 10000
h = (tn - t0)/N
t_eval = np.linspace(t0, tn, N + 1)

sigma = 10
rho = 28
beta = 8/3


# Solve using RK45
def rk45(t_span, init_state):
    sol = solve_ivp(lorenz, t_span, init_state, method='RK45',
                    args=(sigma, rho, beta), t_eval=t_eval)
    return sol.t, sol.y.T


t_rk45, sol_rk45 = rk45(t_span, init_state)
x_rk45, y_rk45, z_rk45 = sol_rk45[:, 0], sol_rk45[:, 1], sol_rk45[:, 2]


# Calculate the energy for RK45
E = (x_rk45**2) / 2 + y_rk45**2 + z_rk45**2 - z_rk45


# Calculate the exponential decay
V0 = 1.0
decay_rate = sigma + 1 + beta
V = V0*np.exp(-decay_rate*t_rk45)


# Plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(t_rk45, E, lw=0.4)
ax1.set_title('Lorenz energy for RK45')
ax1.set_xlabel('time')
ax1.set_ylabel('Lorenz energy E(t)')
ax1.grid(True)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(t_rk45, V, lw=1.5)
ax2.set_title('Exponential decay for volume')
ax2.set_xlabel('time')
ax2.set_ylabel('Volume V(t)')
ax2.grid(True)

plt.tight_layout()
plt.show()
