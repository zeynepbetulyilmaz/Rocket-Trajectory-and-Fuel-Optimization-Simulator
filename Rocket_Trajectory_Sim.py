import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d, CubicSpline
from scipy.linalg import lu_factor, lu_solve
import tkinter as tk
from tkinter import messagebox

# -----------------------------
# Constants
# -----------------------------
g = 9.81                     # Gravity (m/s^2)
Cd = 0.5                     # Drag coefficient
rho = 1.225                  # Air density (kg/m^3)
A = 0.05                     # Cross-sectional area (m^2)
m0 = 50.0                   # Initial mass (kg)
mf = 20.0                   # Final mass (kg)
burn_time = 20.0            # Time thrust is applied (s)
T_max = 1000.0              # Max thrust for GUI validation
dt = 0.1                   # Time step for integration
t_max = 60.0                # Total simulation time (s)

# -----------------------------
# Physics Functions
# -----------------------------
def thrust_profile(t, T_peak, burn_time):
    return T_peak if t <= burn_time else 0

def drag_force(v):
    return 0.5 * Cd * rho * A * v**2 * np.sign(v)

def mass_profile(t, m0, mf, burn_time):
    return mf if t > burn_time else m0 - (m0 - mf) * (t / burn_time)

def rocket_dynamics(t, y, T_peak):
    v, h = y
    m = mass_profile(t, m0, mf, burn_time)
    T = thrust_profile(t, T_peak, burn_time)
    D = drag_force(v)
    a = (T - D - m * g) / m
    return np.array([a, v])

# -----------------------------
# Numerical Methods
# -----------------------------
def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt * k1 / 2, *args)
    k3 = f(t + dt/2, y + dt * k2 / 2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate(T_peak):
    t_values = np.arange(0, t_max + dt, dt)
    y = np.array([0.0, 0.0])  # Initial velocity and height
    trajectory = []
    for t in t_values:
        trajectory.append((t, y[0], y[1], mass_profile(t, m0, mf, burn_time)))
        y = rk4_step(rocket_dynamics, t, y, dt, T_peak)
    return np.array(trajectory)

def numerical_diff(f, h):
    fwd = (f[2:] - f[1:-1]) / h
    bwd = (f[1:-1] - f[:-2]) / h
    cent = (f[2:] - f[:-2]) / (2 * h)
    return fwd, bwd, cent

def trapezoidal_rule(y, h):
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# -----------------------------
# GUI Simulation Handler
# -----------------------------
def run_simulation_gui():
    try:
        thrust_input = float(thrust_entry.get())
        if not (100.0 <= thrust_input <= T_max):
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", f"Enter a thrust value between 100 and {T_max} N.")
        return

    data = simulate(thrust_input)
    times = data[:, 0]
    velocities = data[:, 1]
    altitudes = data[:, 2]
    masses = data[:, 3]

    cent = numerical_diff(altitudes, dt)[2]
    total_altitude = trapezoidal_rule(velocities, dt)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    axs[0].plot(times, altitudes); axs[0].set_title("Altitude vs Time"); axs[0].grid()
    axs[1].plot(times, velocities, color="orange"); axs[1].set_title("Velocity vs Time"); axs[1].grid()
    axs[2].plot(times, masses, color="green"); axs[2].set_title("Mass vs Time"); axs[2].grid()
    axs[3].plot(times[1:-1], cent, color="purple"); axs[3].set_title("Central Diff. of Altitude"); axs[3].grid()

    plt.tight_layout()
    plt.show()

    messagebox.showinfo("Simulation Result", f"Total altitude (by trapezoidal rule): {total_altitude:.2f} m")

# -----------------------------
# Tkinter GUI
# -----------------------------
window = tk.Tk()
window.title("Rocket Trajectory Simulator")
window.geometry("400x200")

label = tk.Label(window, text="Enter Thrust (100 - 1000 N):")
label.pack(pady=10)

thrust_entry = tk.Entry(window, width=20)
thrust_entry.pack()
thrust_entry.insert(0, "500")

simulate_button = tk.Button(window, text="Run Simulation", command=run_simulation_gui)
simulate_button.pack(pady=20)

window.mainloop()
