import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# This code is to compare the RK4 and RK45 further

# Define lorenz equations
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma*(y-x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return np.array([dxdt, dydt, dzdt])


# Set initial condition, the parameters, and time steps
init_state = [1.0, 1.0, 1.0]

t_span = (0, 200)
t0, tn = t_span
N = 20000
h = (tn - t0)/N
t_eval = np.linspace(t0, tn, N + 1)

sigma = 10
rho = 28
beta = 8/3


# RK4
def rk4(init_state, t_span, N):
    x = np.empty(N + 1)
    y = np.empty(N + 1)
    z = np.empty(N + 1)

    state = np.array(init_state)
    x[0], y[0], z[0] = init_state

    for i in range(N):
        t = t_eval[i]
        k1 = lorenz(t, state, sigma, rho, beta)
        k2 = lorenz(t + h/2, state + h/2 * k1, sigma, rho, beta)
        k3 = lorenz(t + h/2, state + h/2 * k2, sigma, rho, beta)
        k4 = lorenz(t + h, state + h * k3, sigma, rho, beta)
        state = state + h/6 * (k1 + 2*k2 + 2*k3 + k4)

        x[i+1], y[i+1], z[i+1] = state
    return t_eval, x, y, z


t_rk4, x_rk4, y_rk4, z_rk4 = rk4(init_state, t_span, N)


# RK45
def rk45(t_span, init_state):
    sol = solve_ivp(lorenz, t_span, init_state, method='RK45',
                    args=(sigma, rho, beta), t_eval=t_eval)
    return sol.t, sol.y.T


t_rk45, sol_rk45 = rk45(t_span, init_state)
x_rk45, y_rk45, z_rk45 = sol_rk45[:, 0], sol_rk45[:, 1], sol_rk45[:, 2]


# Long-term accuracy analysis
fig = plt.figure(figsize=(20, 15))

# Time series comparison - full range
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t_rk4, x_rk4, 'b-', lw=0.5, label='RK4')
ax1.plot(t_rk45, x_rk45, 'r-', lw=0.5, label='RK45')
ax1.set_title('x vs. t: Full Time Range [0, 200]')
ax1.set_xlabel('time')
ax1.set_ylabel('x')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Focus on later time behavior [150, 200]
late_mask = t_rk4 >= 170
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_rk4[late_mask], x_rk4[late_mask], 'b-', lw=0.8, label='RK4')
ax2.plot(t_rk45[late_mask], x_rk45[late_mask], 'r-', lw=0.8, label='RK45')
ax2.set_title('x vs. t: Late Time [170, 200]')
ax2.set_xlabel('time')
ax2.set_ylabel('x')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Phase space comparison - full trajectory
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x_rk4, y_rk4, 'b-', lw=0.3, label='RK4')
ax3.plot(x_rk45, y_rk45, 'r-', lw=0.3, label='RK45')
ax3.set_title('Phase Space: x-y (Full Trajectory)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Phase space - late time only
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x_rk4[late_mask], y_rk4[late_mask], 'b-', lw=0.5, label='RK4')
ax4.plot(x_rk45[late_mask], y_rk45[late_mask], 'r-', lw=0.5, label='RK45')
ax4.set_title('Phase Space: x-y (Late Time [170, 200])')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()
