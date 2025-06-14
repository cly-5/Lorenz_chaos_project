#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters ---
def energy(x, y, px, py):
    return 0.5 * (px**2 + py**2 + x**2 + y**2) + x**2 * y - (1/3) * y**3

def leapfrog_step(x, y, px, py, dt):
    # Half-step momentum
    px_half = px - 0.5 * dt * (x + 2 * x * y)
    py_half = py - 0.5 * dt * (y + x**2 - y**2)

    # Full-step position
    x_new = x + dt * px_half
    y_new = y + dt * py_half

    # Full-step momentum
    px_new = px_half - 0.5 * dt * (x_new + 2 * x_new * y_new)
    py_new = py_half - 0.5 * dt * (y_new + x_new**2 - y_new**2)

    return x_new, y_new, px_new, py_new

# --- Initial Conditions ---
x, y = 0.1, 0.0
px, py = 0.0, 0.3
state = [x, y, px, py]

dt = 0.01
n_steps = 50000

# --- Data Storage ---
time_list = []
energy_list = []
poincare_px = []
poincare_x = []

# --- Simulation Loop ---
for step in range(n_steps):
    x, y, px, py = state
    x, y, px, py = leapfrog_step(x, y, px, py, dt)
    state = [x, y, px, py]

    # Energy
    E = energy(x, y, px, py)
    energy_list.append(E)
    time_list.append(step * dt)

    # Poincaré section: when y ≈ 0 and py > 0
    if abs(y) < 1e-3 and py > 0:
        poincare_x.append(x)
        poincare_px.append(px)

# --- Plotting ---
plt.figure(figsize=(12, 5))

# Energy plot
plt.subplot(1, 2, 1)
plt.plot(time_list, energy_list)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy Conservation (Leapfrog)")
plt.grid(True)

# Poincaré section
plt.subplot(1, 2, 2)
plt.plot(poincare_x, poincare_px, '.', markersize=1)
plt.xlabel("x")
plt.ylabel("px")
plt.title("Poincaré Section (y ≈ 0, py > 0)")
plt.grid(True)

plt.tight_layout()
plt.show()


# In[1]:


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def henon_heiles(t, state, E):
    x, y, px, py = state
    dxdt = px
    dydt = py
    dpxdt = -x - 2*x*y
    dpydt = -y - x**2 + y**2
    return [dxdt, dydt, dpxdt, dpydt]


def initial_momentum(x, y, E):
    V = 0.5*(x**2 + y**2) + x**2*y - (1/3)*y**3  
    T = E - V  
    if T < 0:
        return None 
    py = np.sqrt(2*T)  
    return [x, y, 0, py]


E_levels = [0.08, 0.12, 0.17] 
ics_low = [(0.1, 0), (0.2, 0)] 
ics_high = [(0.3, 0), (0.4, 0)] 
t_span = (0, 100) 


fig = plt.figure(figsize=(12, 8), dpi=150)
gs = GridSpec(2, 3, figure=fig)
axs = [
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
]


for i, E in enumerate(E_levels):
    for j, (x0, y0) in enumerate(ics_low + ics_high):
        state0 = initial_momentum(x0, y0, E)
        if state0 is None:
            print(f"no solution under ({x0},{y0})")
            continue
        
        sol = solve_ivp(henon_heiles, t_span, state0, args=(E,), 
                       method='RK45', rtol=1e-8, atol=1e-8)
        

        row = 0 if j < 2 else 1
        ax = axs[row][i]
        

        ax.plot(sol.y[0], sol.y[1], lw=0.5, 
               label=f'$(x_0,y_0)=({x0},{y0})$')
        ax.scatter(x0, y0, c='r', s=10) 
        

        ax.set_title(f'$E={E}$', fontsize=10)
        ax.set_xlabel('$x$', fontsize=9)
        ax.set_ylabel('$y$', fontsize=9)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.5, 0.6)
        ax.grid(alpha=0.3)
        if i == 2 and row == 1:
            ax.legend(fontsize=7, loc='upper right')

plt.suptitle('Hénon-Heiles System: Trajectories at Different Energies', y=0.95)
plt.tight_layout()
plt.savefig('henon_heiles_trajectories.png', bbox_inches='tight')
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def henon_heiles(t, u):
    """Henon-Heiles system equations of motion"""
    x, y, px, py = u
    dxdt = px
    dydt = py
    dpxdt = -x - 2*x*y
    dpydt = -y - (x**2 - y**2)
    return [dxdt, dydt, dpxdt, dpydt]

def find_initial_conditions(y0, E=1/8):
    """Find initial px given y0 to satisfy energy E=1/8"""
    # Energy equation with x0=0, py0=0: E = 0.5*px0^2 + 0.5*y0^2 - y0^3/3
    px0_squared = 2*(E - 0.5*y0**2 + y0**3/3)
    if px0_squared < 0:
        return None  # No real solution for this y0
    px0 = np.sqrt(px0_squared)
    return [0, y0, px0, 0]  # [x0, y0, px0, py0]

def compute_trajectory(initial_conditions, t_max=200):
    """Compute trajectory for given initial conditions"""
    sol = solve_ivp(henon_heiles, [0, t_max], initial_conditions,
                   rtol=1e-8, atol=1e-8, dense_output=True)
    return sol

# Create 9 y vs x plots (3x3 grid)
y0_values = np.linspace(-0.4, 0.6, 9)  # 9 y0 values from -0.4 to 0.6
t_max = 500  # Sufficient integration time to see the trajectory

plt.figure(figsize=(15, 7.5))

for i, y0 in enumerate(y0_values, 1):
    initial_conditions = find_initial_conditions(y0)
    if initial_conditions is None:
        print(f"No solution for y0 = {y0:.3f} (energy condition not satisfied)")
        continue
        
    sol = compute_trajectory(initial_conditions, t_max)
    
    # Create subplot (y vs x)
    plt.subplot(2, 5, i)
    plt.plot(sol.y[0], sol.y[1], ',', markersize=1, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Initial y0 = {y0:.2f}')
    plt.grid(True)
    plt.axis('equal')  # Keep aspect ratio equal

plt.tight_layout()
plt.suptitle('Henon-Heiles Position Space (y vs x) for E=1/8, x0=0, py0=0', y=1.02)
plt.show()


# In[36]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time  # For timing computations

# =============================================
# Hénon-Heiles System
# =============================================
def henon_heiles(t, y):
    q1, q2, p1, p2 = y
    dq1 = p1
    dq2 = p2
    dp1 = -q1 - 2 * q1 * q2
    dp2 = -q2 - (q1**2 - q2**2)
    return np.array([dq1, dq2, dp1, dp2])

def hamiltonian(y):
    q1, q2, p1, p2 = y
    return 0.5 * (p1**2 + p2**2) + 0.5 * (q1**2 + q2**2) + q1**2 * q2 - (1/3) * q2**3


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def symplectic_euler_step(f, t, y, h):
    q1, q2, p1, p2 = y
    # Update momenta first (using current positions)
    p1_new = p1 + h * (-q1 - 2 * q1 * q2)
    p2_new = p2 + h * (-q2 - (q1**2 - q2**2))
    # Update positions (using new momenta)
    q1_new = q1 + h * p1_new
    q2_new = q2 + h * p2_new
    return np.array([q1_new, q2_new, p1_new, p2_new])

def taylor17_step(f, t, y, h):
    # Placeholder: Using DOP853 for high accuracy (true Taylor17 needs AD)
    sol = solve_ivp(f, [t, t + h], y, method='DOP853', rtol=1e-14, atol=1e-14)
    return sol.y[:, -1]

y0 = np.array([0.1, 0.0, 0.0, 0.2]) 
t_span = (0, 500)
h = 0.01
steps = int((t_span[1] - t_span[0]) / h)

start_time = time.time()
sol_ref = solve_ivp(henon_heiles, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-12)
ref_time = time.time() - start_time
t_ref = sol_ref.t
y_ref = sol_ref.y


t = np.linspace(t_span[0], t_span[1], steps + 1)
y_rk4 = np.zeros((4, steps + 1))
y_symp_euler = np.zeros((4, steps + 1))
y_taylor = np.zeros((4, steps + 1))

y_rk4[:, 0] = y0
y_symp_euler[:, 0] = y0
y_taylor[:, 0] = y0

# Time the integrations
# RK4
start_time = time.time()
for i in range(steps):
    y_rk4[:, i+1] = rk4_step(henon_heiles, t[i], y_rk4[:, i], h)
rk4_time = time.time() - start_time


start_time = time.time()
for i in range(steps):
    y_symp_euler[:, i+1] = symplectic_euler_step(henon_heiles, t[i], y_symp_euler[:, i], h)
symp_euler_time = time.time() - start_time

# Taylor
start_time = time.time()
for i in range(steps):
    y_taylor[:, i+1] = taylor17_step(henon_heiles, t[i], y_taylor[:, i], h)
taylor_time = time.time() - start_time

# =============================================
# Compute Errors and Statistics
# =============================================
# Interpolate reference solution
y_ref_interp = np.array([np.interp(t, t_ref, y_ref[i, :]) for i in range(4)])

# State error (L2 norm)
error_rk4 = np.linalg.norm(y_rk4 - y_ref_interp, axis=0)
error_symp_euler = np.linalg.norm(y_symp_euler - y_ref_interp, axis=0)
error_taylor = np.linalg.norm(y_taylor - y_ref_interp, axis=0)

# Energy error
H0 = hamiltonian(y0)
H_rk4 = np.array([hamiltonian(y_rk4[:, i]) for i in range(steps + 1)])
H_symp_euler = np.array([hamiltonian(y_symp_euler[:, i]) for i in range(steps + 1)])
H_taylor = np.array([hamiltonian(y_taylor[:, i]) for i in range(steps + 1)])

energy_error_rk4 = np.abs(H_rk4 - H0)
energy_error_symp_euler = np.abs(H_symp_euler - H0)
energy_error_taylor = np.abs(H_taylor - H0)

# Compute mean errors
mean_state_error_rk4 = np.mean(error_rk4)
mean_state_error_symp_euler = np.mean(error_symp_euler)
mean_state_error_taylor = np.mean(error_taylor)

mean_energy_error_rk4 = np.mean(energy_error_rk4)
mean_energy_error_symp_euler = np.mean(energy_error_symp_euler)
mean_energy_error_taylor = np.mean(energy_error_taylor)

# =============================================
# Print Performance Comparison
# =============================================
print("\n=== Performance Comparison ===")
print(f"{'Method':<20} {'Time (s)':<12} {'Mean State Error':<18} {'Mean Energy Error':<18}")
print(f"{'Reference':<20} {ref_time:<12.4f} {'-':<18} {'-':<18}")
print(f"{'RK4':<20} {rk4_time:<12.4f} {mean_state_error_rk4:<18.4e} {mean_energy_error_rk4:<18.4e}")
print(f"{'Taylor (Order 17)':<20} {taylor_time:<12.4f} {mean_state_error_taylor:<18.4e} {mean_energy_error_taylor:<18.4e}")

# =============================================
# Plot Results
# =============================================
plt.figure(figsize=(18, 5))

# State Error
plt.subplot(1, 3, 1)
plt.plot(t, error_rk4, label=f"RK4 (Avg: {mean_state_error_rk4:.2e})", color='red')
plt.plot(t, error_taylor, label=f"Taylor (Avg: {mean_state_error_taylor:.2e})", color='green')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('State Error (L2 norm)')
plt.legend()
plt.title("State Error Comparison")

# Energy Error
plt.subplot(1, 3, 2)
plt.plot(t, energy_error_rk4, label=f"RK4 (Avg: {mean_energy_error_rk4:.2e})", color='red')
plt.plot(t, energy_error_taylor, label=f"Taylor (Avg: {mean_energy_error_taylor:.2e})", color='green')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Energy Error |H(t) - H(0)|')
plt.legend()
plt.title("Energy Conservation")

# Computational Time Bar Plot
plt.subplot(1, 3, 3)
methods = ['RK4', 'Taylor\n(Order 17)']
times = [rk4_time, taylor_time]
plt.bar(methods, times, color=['red', 'green'])
plt.ylabel('Computation Time (s)')
plt.title("Computational Cost Comparison")

plt.tight_layout()
plt.show()


# In[ ]:




