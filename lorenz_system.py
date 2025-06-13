import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Simulate and plot the Lorenz system using the methods

# Define lorenz equations
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma*(y-x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return np.array([dxdt, dydt, dzdt])


# Set initial condition, the parameters, and time steps
init_state = [1.0, 1.0, 1.0]

t_span = (0, 40)
t0, tn = t_span
N = 10000
h = (tn - t0)/N
t_eval = np.linspace(t0, tn, N + 1)

sigma = 10
rho = 28
beta = 8/3


# Euler
def euler(init_state, t_span, N):
    x = np.empty(N + 1)
    y = np.empty(N + 1)
    z = np.empty(N + 1)

    x[0], y[0], z[0] = init_state
    state = np.array(init_state)

    for i in range(N):
        t = t_eval[i]
        dt = lorenz(t, state, sigma, rho, beta)
        state = state + h * dt
        x[i + 1], y[i + 1], z[i + 1] = state

    return t_eval, x, y, z


t_e, x_e, y_e, z_e = euler(init_state, t_span, N)


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


# Plot
fig = plt.figure(figsize=(15, 10))

# Plot Euler solution
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot(x_e, y_e, z_e, 'g-', lw=0.5)
ax1.set_title('Euler Method')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Plot RK4
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot(x_rk4, y_rk4, z_rk4, 'b-', lw=0.5)
ax2.set_title('RK4 Method')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# Plot RK45
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot(x_rk45, y_rk45, z_rk45, 'r-', lw=0.5)
ax3.set_title('RK45 Method')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

plt.tight_layout()
plt.show()


# Time Plots
fig = plt.figure(figsize=(15, 6))


# Plot Euler x v.t
ax11 = fig.add_subplot(3, 3, 1)
ax11.plot(t_e, x_e, 'g-', lw=0.5)
ax11.set_title('Euler Method: x vs. t')
ax11.set_xlabel('time')
ax11.set_ylabel('x')
ax11.grid(True, alpha=0.3)

# Plot Euler y v.t
ax12 = fig.add_subplot(3, 3, 2)
ax12.plot(t_e, y_e, 'g-', lw=0.5)
ax12.set_title('Euler Method: y vs. t')
ax12.set_xlabel('time')
ax12.set_ylabel('y')
ax12.grid(True, alpha=0.3)

# Plot Euler z v.t
ax13 = fig.add_subplot(3, 3, 3)
ax13.plot(t_e, z_e, 'g-', lw=0.5)
ax13.set_title('Euler Method: z vs. t')
ax13.set_xlabel('time')
ax13.set_ylabel('z')
ax13.grid(True, alpha=0.3)

# Plot RK4 x v. t
ax21 = fig.add_subplot(3, 3, 4)
ax21.plot(t_rk4, x_rk4, 'b-', lw=0.5)
ax21.set_title('RK4 Method: x vs. t')
ax21.set_xlabel('time')
ax21.set_ylabel('x')
ax21.grid(True, alpha=0.3)

# Plot RK4 y v. t
ax22 = fig.add_subplot(3, 3, 5)
ax22.plot(t_rk4, y_rk4, 'b-', lw=0.5)
ax22.set_title('RK4 Method: y vs. t')
ax22.set_xlabel('time')
ax22.set_ylabel('y')
ax22.grid(True, alpha=0.3)

# Plot RK4 z v. t
ax23 = fig.add_subplot(3, 3, 6)
ax23.plot(t_rk4, z_rk4, 'b-', lw=0.5)
ax23.set_title('RK4 Method: z vs. t')
ax23.set_xlabel('time')
ax23.set_ylabel('z')
ax23.grid(True, alpha=0.3)

# Plot RK45 x v. t
ax31 = fig.add_subplot(3, 3, 7)
ax31.plot(t_rk45, x_rk45, 'r-', lw=0.5)
ax31.set_title('RK45 Method: x vs. t')
ax31.set_xlabel('time')
ax31.set_ylabel('x')
ax31.grid(True, alpha=0.3)

# Plot RK45 y v. t
ax32 = fig.add_subplot(3, 3, 8)
ax32.plot(t_rk45, y_rk45, 'r-', lw=0.5)
ax32.set_title('RK45 Method: y vs. t')
ax32.set_xlabel('time')
ax32.set_ylabel('y')
ax32.grid(True, alpha=0.3)

# Plot RK45 z v. t
ax33 = fig.add_subplot(3, 3, 9)
ax33.plot(t_rk45, z_rk45, 'r-', lw=0.5)
ax33.set_title('RK45 Method: z vs. t')
ax33.set_xlabel('time')
ax33.set_ylabel('z')
ax33.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
