import numpy as np
from functools import cache
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

init_conds = [
    {
        "name": "anim1",
        "m1": 1,
        "x1": -5,
        "y1": 0,
        "vx1": 1,
        "vy1": 0,
        "m2": 1,
        "x2": 5,
        "y2": 0,
        "vx2": -1,
        "vy2": 0,
    },
    {
        "name": "anim2",
        "m1": 1,
        "x1": -5,
        "y1": 0,
        "vx1": 1,
        "vy1": 0,
        "m2": 1,
        "x2": 5,
        "y2": 1,
        "vx2": -1,
        "vy2": 0,
    },
]

def render_particles(paths, t, title, xlabel, ylabel):
    num_particles = len(paths)
    available_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.grid(True)

    colors = random.sample(available_colors, num_particles)
    particles = [
        ax.plot([], [], "o", color=c, markersize=8, label=f"Particle {i+1}")[0]
        for i, c in enumerate(colors)
    ]
    trails = [ax.plot([], [], "-", color=c, alpha=0.3)[0] for c in colors]

    ax.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    def init():
        for particle, trail in zip(particles, trails):
            particle.set_data([], [])
            trail.set_data([], [])
        return particles + trails

    def update(frame):
        path_length = 30
        for i, (particle, trail) in enumerate(zip(particles, trails)):
            particle.set_data([paths[i][frame, 0]], [paths[i][frame, 1]])
            start = max(0, frame - path_length)
            trail.set_data(
                paths[i][start : frame + 1, 0],
                paths[i][start : frame + 1, 1],
            )
        return particles + trails

    anim = FuncAnimation(
        fig, update, frames=len(t), init_func=init, blit=True, interval=50, repeat=True
    )

    anim.save(f"{title}.gif", writer="ffmpeg", fps=20)

    plt.show()
    return anim 

def lenard_jones_potential(r: float) -> float:
    return 24 * (2 / r**13 - 1 / r**7)

def system(t, y, m1, m2):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    a = lenard_jones_potential(r)
    a1 = a / m1
    a2 = a / m2
    ax1 = (x2 - x1) * a1
    ay1 = (y2 - y1) * a1
    ax2 = (x1 - x2) * a2
    ay2 = (y1 - y2) * a2
    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

if __name__ == "__main__":
    for condition in init_conds:
        y0 = [
            condition["x1"],
            condition["y1"],
            condition["x2"],
            condition["y2"],
            condition["vx1"],
            condition["vy1"],
            condition["vx2"],
            condition["vy2"],
        ]
        t_end = 10
        t_span = [0, 10]
        t_eval = np.linspace(0, t_end, 100)
        solution = solve_ivp(
            system,
            t_span,
            y0,
            t_eval=t_eval,
            args=(condition["m1"], condition["m2"]),
        )
        trajectory_1 = np.array([solution.y[0], solution.y[1]]).T
        trajectory_2 = np.array([solution.y[2], solution.y[3]]).T
        anim = render_particles(
            [trajectory_1, trajectory_2],
            solution.t,
            condition["name"],
            "x(t)",
            "y(t)",
        )