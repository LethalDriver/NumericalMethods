import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Gravitational constant
G = 1

y0 = [
    -1.0, 0.0,               # x1, y1
    1.0, 0.0,                # x2, y2
    0.0, 2.0,               # x3, y3
    0.0, -0.5,               # vx1, vy1
    0.0, 0.5,                # vx2, vy2
    1.0, 0.0                 # vx3, vy3
]


def system(t, y):
    # Unpack positions and velocities
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y

    # Distances between planets
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    # Accelerations due to gravity
    ax1 = G * ((x2 - x1) / r12**3 + (x3 - x1) / r13**3)
    ay1 = G * ((y2 - y1) / r12**3 + (y3 - y1) / r13**3)
    ax2 = G * ((x1 - x2) / r12**3 + (x3 - x2) / r23**3)
    ay2 = G * ((y1 - y2) / r12**3 + (y3 - y2) / r23**3)
    ax3 = G * ((x1 - x3) / r13**3 + (x2 - x3) / r23**3)
    ay3 = G * ((y1 - y3) / r13**3 + (y2 - y3) / r23**3)

    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]



# Time span
t_end = 20
t_span = [0, t_end]
t_eval = np.linspace(0, t_end, 1000)

# Solve the ODE
solution = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

def animate_particles():
    # Extract positions from the solution
    x1 = solution.y[0]
    y1 = solution.y[1]
    x2 = solution.y[2]
    y2 = solution.y[3]
    x3 = solution.y[4]
    y3 = solution.y[5]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min(min(x1), min(x2), min(x3)), max(max(x1), max(x2), max(x3)))
    ax.set_ylim(min(min(y1), min(y2), min(y3)), max(max(y1), max(y2), max(y3)))
    ax.set_aspect('auto')
    ax.grid(True)

    # Initialize lines for the planets
    line1, = ax.plot([], [], 'o-', color='red', markersize=1, label='Planet 1')
    line2, = ax.plot([], [], 'o-', color='blue', markersize=1, label='Planet 2')
    line3, = ax.plot([], [], 'o-', color='green', markersize=1, label='Planet 3')

    plt.title('Three-Body Problem')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    ax.legend()

    def init():
        """Initialize animation"""
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def update(frame):
        """Update animation"""
        line1.set_data(x1[:frame], y1[:frame])
        line2.set_data(x2[:frame], y2[:frame])
        line3.set_data(x3[:frame], y3[:frame])
        return line1, line2, line3

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(t_eval),
                         init_func=init, blit=True,
                         interval=20, repeat=True)

    plt.show()
    return anim  # Keep a reference to prevent garbage collection

# Run the animation
if __name__ == "__main__":
    anim = animate_particles()