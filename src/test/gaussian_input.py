"""
Plot Gaussian bumps on a circle and draw circle patterns with optional fixation point.
This script is intended for paper figure generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

temp_dir = Path(__file__).resolve().parents[2] / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

# colors_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]
colors_list = ["#16A34A", "#D4B106", "#F97316", "#C93F3F", "#A855F7", "#2563EB"]


def darken_color(hex_color, factor=0.6):
    """Darken a hex color by multiplying RGB values by factor (0-1)."""
    rgb = mcolors.to_rgb(hex_color)
    dark_rgb = tuple(max(0, c * factor) for c in rgb)
    return mcolors.to_hex(dark_rgb)


# Generate darker edge colors
dark_edge_colors = [darken_color(color) for color in colors_list]

def plot_gaussian_on_circle(plot_gaussian_bumps=False, chosen_idx=None):
    N = 24
    target_indices = [0, 4, 8, 12, 16, 20]

    R = 4.2  # radius of the circle

    sigma = 0.5  # standard deviation of Gaussian bumps
    amplitude = 1.2  # height of Gaussian bumps
    gap = 0.8  # distance from circle to the center of Gaussian bumps
    t_range = 1.5  # range along the tangent direction

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_points = R * np.cos(angles)
    y_points = R * np.sin(angles)

    fig, ax = plt.subplots(figsize=(12, 12))

    # gray dots for all 24 points
    for idx in range(N):
        if idx in target_indices:
            continue
        ax.scatter(x_points[idx], y_points[idx], c="gray", s=400)

    counter = 0
    for idx in target_indices:
        theta = angles[idx]
        cx = x_points[idx]
        cy = y_points[idx]

        # colored dot for item points
        if chosen_idx is not None:
            alpha = 1.0 if counter == chosen_idx else 0.5
        else:
            alpha = 1.0

        ax.plot(
            cx,
            cy,
            marker="o",
            color=colors_list[counter],
            markersize=108,
            zorder=10,
            alpha=alpha,
        )

        t = np.linspace(-t_range, t_range, 100)
        radial_offset = amplitude * np.exp(-(t**2) / (2 * sigma**2))

        vec_radial = np.array([np.cos(theta), np.sin(theta)])
        vec_tangent = np.array([-np.sin(theta), np.cos(theta)])

        curve_x = cx + (gap + radial_offset) * vec_radial[0] + t * vec_tangent[0]
        curve_y = cy + (gap + radial_offset) * vec_radial[1] + t * vec_tangent[1]

        if plot_gaussian_bumps:
            ax.plot(curve_x, curve_y, "b-", linewidth=4, zorder=5, alpha=0.8)

        counter += 1

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    # plt.tight_layout(pad=8.0)
    fig.savefig(f"{temp_dir}/gaussian_circle_chosen_{chosen_idx}.png", dpi=600)
    print(f"figure saved to {temp_dir}/gaussian_on_circle_chosen_idx_{chosen_idx}.png")
    plt.close(fig)
    # plt.show()


def draw_pattern(fixation_enabled=True, active_index=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    bg_color = "#ffffff"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    radius = 1.0

    angles_deg = np.array(60 * np.arange(6))
    angles_rad = np.radians(angles_deg)

    x_coords = radius * np.cos(angles_rad)
    y_coords = radius * np.sin(angles_rad)

    # plot fixation point if enabled
    if fixation_enabled:
        ax.scatter(0, 0, s=1000, c="black", marker="D", zorder=10, edgecolors="none")

    circle_size = 3000

    for i in range(6):
        x, y = x_coords[i], y_coords[i]

        if active_index is not None and i == active_index:
            ax.scatter(
                x,
                y,
                s=circle_size,
                c=colors_list[active_index],
                edgecolors="none",
                zorder=5,
            )
        else:
            ax.scatter(
                x,
                y,
                s=circle_size,
                c=bg_color,
                edgecolors="black",
                linewidths=4,
                zorder=5,
            )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(
        f"{temp_dir}/circle_pattern_{active_index}_fixation={fixation_enabled}.png", dpi=600
    )
    print(
        f"figure saved to {temp_dir}/circle_pattern_{active_index}_fixation={fixation_enabled}.png"
    )
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_gaussian_on_circle()
    for i in range(6):
        plot_gaussian_on_circle(chosen_idx=i)

    draw_pattern()
    for i in range(6):
        draw_pattern(fixation_enabled=True, active_index=i)
        draw_pattern(fixation_enabled=False, active_index=i)
