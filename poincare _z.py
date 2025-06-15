import numpy as np
import matplotlib.pyplot as plt


def lorenz(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def rk4_step(f, state, dt):
    k1 = f(state)
    k2 = f(state + dt * k1 / 2)
    k3 = f(state + dt * k2 / 2)
    k4 = f(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def compute_poincare(initial_state, dt=0.01, t_max=100.0, z_section=25.0, transient=10.0):
    num_steps = int(t_max / dt)
    state = np.array(initial_state, dtype=float)
    t = 0.0
    poincare_x = []
    poincare_y = []
    for _ in range(num_steps):
        prev_state = state.copy()
        state = rk4_step(lorenz, state, dt)
        t += dt
        if t > transient and prev_state[2] < z_section <= state[2]:
            alpha = (z_section - prev_state[2]) / (state[2] - prev_state[2])
            x_cross = prev_state[0] + alpha * (state[0] - prev_state[0])
            y_cross = prev_state[1] + alpha * (state[1] - prev_state[1])
            poincare_x.append(x_cross)
            poincare_y.append(y_cross)
    return poincare_x, poincare_y


if __name__ == "__main__":
    initial_conditions = [(1, 1, 1), (10, 10, 10), (30, 30, 30)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, ic in zip(axes, initial_conditions):
        x_cross, y_cross = compute_poincare(ic)
        ax.scatter(x_cross, y_cross, s=1)
        ax.set_title(f'Initial = {ic}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
