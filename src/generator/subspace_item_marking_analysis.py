"""
使用不同rank的item标签来标记数据以验证假设。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

import generator.utils as utils

#### parameters
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name
params = utils.initialize_analysis(TASK, N_SAMPLES, FILENAME)

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

hiddens = outputs_dict["hiddens"]
hiddens = np.tanh(hiddens)

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"] - 10, :]

delay_mean = np.mean(hiddens_delay, axis=1)
pca_x = PCA(n_components=8).fit(delay_mean)
Low = pca_x.transform(delay_mean)

plt.rcParams["font.sans-serif"] = "Arial"
LABEL_FONTSIZE = 18
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


def compute_one(class_i):
    class_i = class_i % N_CLASS
    item_1, item_2, item_3 = PERMS[class_i]

    if TASK == "backward":
        return item_1, item_2, item_3
    elif TASK == "forward":
        return item_1, item_2, item_3
    elif TASK == "forward2backward":
        return item_3, item_2, item_1
    else:
        raise ValueError("Unknown TASK")


# 构建线性回归模型
# one-hot vector
X1 = []
X2 = []
X3 = []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)

base_colors, colors_rank1, colors_rank2, colors_rank3 = utils.get_color(X1, X2, X3)
colors_empty = ["gray" for _ in range(N_SAMPLES)]


def set_axis_style(ax):
    ax.set_xticks([-6, 0, 6], labels=["-6", "", "6"])
    ax.set_yticks([-6, 0, 6], labels=["", "", "6"])

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.spines["bottom"].set_bounds(-6, 6)
    ax.spines["left"].set_bounds(-6, 6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=1, direction="in", length=4)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.set_aspect("equal", adjustable="box")


def plot_delay_pca_grid(color_rows, fig_name="pca_analysis_delay_stage_grid_all_ranks"):
    """
    Plot a 3x3 grid: rows are rank1/2/3 colorings, columns are PC1-2, PC3-4, PC5-6.
    """

    fig, axes = plt.subplots(3, 3, figsize=(21, 18))

    marker_size = 48

    pc_pairs = [(0, 1), (2, 3), (4, 5)]
    row_titles = ["rank1-labeled", "rank2-labeled", "rank3-labeled"]

    for r in range(3):
        colors_row = color_rows[r]
        for c in range(3):
            ax = axes[r, c]
            p0, p1 = pc_pairs[c]
            ax.scatter(Low[:, p0], Low[:, p1], c=colors_row, marker="o", s=marker_size)

            set_axis_style(ax)

            ax.set_xlabel(f"PC{p0+1}", labelpad=-7)
            ax.set_ylabel(f"PC{p1+1}", labelpad=-6)

    # legend (use the same legend as before)
    # Legend uses the base hue for each item (no shading)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"item {i}",
            markerfacecolor=base_colors[i],
            markersize=8,
        )
        for i in range(6)
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.96)
    )
    fig.suptitle(
        f"Delay PCA of {TASK} task colored by different stimulus items (rows: ranks, cols: PC pairs)",
        fontsize=18,
    )
    # adjust spacing and leave room on the left for large row titles
    plt.subplots_adjust(left=0.12, wspace=0.28, hspace=0.28, top=0.92)

    # Add larger, centered row titles on the left side of the figure
    for r in range(3):
        bbox = axes[r, 0].get_position()
        y = bbox.y0 + bbox.height / 2.0
        fig.text(0.03, y, row_titles[r], va="center", ha="left", fontsize=14)

    out_path = FIG_DIR / f"{TASK}_{fig_name}.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    print(f"Figure saved to {out_path}")
    plt.close()


def plot_delay_different_pc_grid(
    rank, color, fig_name="pca_analysis_delay_stage_different_pc_grid"
):
    """
    Plot a 3x5 grid, traversing different PC pairs in each row.
    """
    # expected: Low has shape (N_SAMPLES, n_components) and n_components >= 8
    n_components = 6

    # Create list of all unique PC pairs (i<j)
    pc_pairs = []
    for i in range(n_components):
        for j in range(i + 1, n_components):
            pc_pairs.append((i, j))

    # We need 15 subplots (3 rows x 5 cols)
    total_plots = 15
    # If there are fewer than total_plots unique pairs, repeat from start to fill
    if len(pc_pairs) < total_plots:
        # repeat the sequence as needed
        times = (total_plots + len(pc_pairs) - 1) // len(pc_pairs)
        pc_pairs = (pc_pairs * times)[:total_plots]
    else:
        pc_pairs = pc_pairs[:total_plots]

    nrows, ncols = 5, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 25))

    marker_size = 36

    for idx, (p0, p1) in enumerate(pc_pairs):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        ax.scatter(Low[:, p0], Low[:, p1], c=color, s=marker_size, marker="o")

        set_axis_style(ax)

        ax.set_xlabel(f"PC{p0+1}", labelpad=-12)
        ax.set_ylabel(f"PC{p1+1}", labelpad=-10)

    # turn off any unused axes (if any)
    for j in range(total_plots, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.suptitle(
        f"Delay PCA projections (rank={rank}) across 15 PC subspaces", fontsize=16
    )
    plt.subplots_adjust(wspace=0.35, hspace=0.35, top=0.93)

    out_path = FIG_DIR / f"{TASK}_{fig_name}_rank{rank}.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    print(f"Figure saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    # produce a single grid figure with three rows (rank1/2/3)
    plot_delay_pca_grid([colors_rank1, colors_rank2, colors_rank3])

    # examine each rank separately
    plot_delay_different_pc_grid(
        rank=1,
        color=colors_rank1,
        fig_name="pca_analysis_delay_stage_different_pc_grid",
    )
    plot_delay_different_pc_grid(
        rank=2,
        color=colors_rank2,
        fig_name="pca_analysis_delay_stage_different_pc_grid",
    )
    plot_delay_different_pc_grid(
        rank=3,
        color=colors_rank3,
        fig_name="pca_analysis_delay_stage_different_pc_grid",
    )
    # plot_delay_different_pc_grid(
    #     rank=1,
    #     color=colors_empty,
    #     fig_name="pca_analysis_delay_stage_different_pc_grid_empty",
    # )
