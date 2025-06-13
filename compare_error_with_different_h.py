import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# this code is to compare the forward Euler and RK45 method for dydt = y
# We aim to see how the error changes as we change the step size h


# Define dydt = y
def func(t, y):
    return y


# Define exact solution of ivp
def sol_exact(t):
    return np.exp(t)


# Set the different N
N_vals = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
h_val = []
errs_e = []
errs_rk4 = []
errs_rk45 = []
t0, tn = 0, 2


# Iterate on N to get values of h
for N in N_vals:
    h = (tn - t0)/N
    h_val.append(h)
    t_eval = np.linspace(t0, tn, N+1)

    # For simplicity we compare the endpoints
    exact_vals = sol_exact(tn)

    # Solve by using forward Euler
    y_euler = np.empty(N + 1)
    y_euler[0] = 1.0

    for i in range(N):
        dydt = func(t_eval[i], y_euler[i])
        y_euler[i+1] = y_euler[i] + h*dydt

    # Euler errors
    err_e = abs(y_euler[-1] - exact_vals)
    errs_e.append(err_e)

    # Solve by RK4
    def rk4(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    y_rk4 = np.empty(N + 1)
    y_rk4[0] = 1.0
    for i in range(N):
        y_rk4[i+1] = rk4(func, t_eval[i], y_rk4[i], h)

    # RK4 errors
    err_rk4 = abs(y_rk4[-1] - exact_vals)
    errs_rk4.append(err_rk4)

    # Solve by RK45
    sol_rk = solve_ivp(func, [t0, tn], [1.0], method='RK45', max_step=h,
                       dense_output=True)
    y_rk45_final = sol_rk.sol(tn)[0]

    # RK45 errors
    err_rk45 = abs(y_rk45_final - exact_vals)
    errs_rk45.append(err_rk45)


# Plot the errors against size step
plt.figure(figsize=(10, 6))
plt.loglog(h_val, errs_e, 'g-', label='Forward Euler')
plt.loglog(h_val, errs_rk4, 'b-', label='RK4')
plt.loglog(h_val, errs_rk45, 'r-', label='RK45')
plt.xlabel('size step (h)')
plt.ylabel('absolute error at endpoint')
plt.title('Errors of numerical methods')
plt.legend()
plt.grid()
plt.show()
