import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

def system(t, y):
    x, v = y
    return [v, 48*(1./x**13 - 0.5/x**7)]

# Initial conditions and time span
y0 = [2, 0]
t_end = 20
t_span = [0, t_end]
t_eval = np.linspace(0, t_end, 1000)

# Solve the ODE
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

def animate_particles():
    # Extract position and velocity from the solution
    x = solution.y[0]
    v = solution.y[1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(v), max(v))
    ax.set_aspect('auto')  # Change aspect ratio to auto
    ax.grid(True)

    # Initialize line for the phase space plot
    line, = ax.plot([], [], 'o-', color='blue', markersize=2)

    plt.title('Phase Space Plot')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v)')

    def init():
        """Initialize animation"""
        line.set_data([], [])
        return line,

    def update(frame):
        """Update animation"""
        line.set_data(x[:frame], v[:frame])
        return line,

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(t_eval),
                         init_func=init, blit=True,
                         interval=20, repeat=True)

    plt.show()
    return anim  # Keep a reference to prevent garbage collection

# Run the animation
if __name__ == "__main__":
    anim = animate_particles()

