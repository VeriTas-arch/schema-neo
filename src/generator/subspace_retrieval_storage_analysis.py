"""
Summary
-------
- 分析retrieval阶段中, 网络如何依次回答三个rank的内容。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
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

# sizeof hiddens is (120, 200, 400), i.e. (batch, T, N_HID)
# batch: N_SAMPLES
hiddens = outputs_dict["hiddens"]
hiddens = np.tanh(hiddens)

# define different time stages
hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"] - 10, :]
hiddens_res1 = hiddens[:, tps["target1_start"] : tps["target1_end"] - 10, :]
hiddens_res2 = hiddens[:, tps["target2_start"] + 2 : tps["target2_end"] - 6, :]
hiddens_res3 = hiddens[:, tps["target3_start"] : tps["target3_end"] - 6, :]

delay_mean = np.mean(hiddens_delay, axis=1)
pca_x = PCA(n_components=8).fit(delay_mean)


def compute_one(class_i):
    class_i = class_i % N_CLASS
    item_1, item_2, item_3 = PERMS[class_i]

    if TASK == "backward":
        return item_1, item_2, item_3
    elif TASK == "forward":
        return item_1, item_2, item_3
        # return item_3, item_2, item_1
    elif TASK == "forward2backward":
        return item_3, item_2, item_1
    elif TASK == "layernorm_backward":
        return item_3, item_2, item_1
    else:
        raise ValueError("Unknown TASK")


# 构建线性回归模型
# one-hot vector
X1, X2, X3 = [], [], []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)

base_colors, colors_rank1, colors_rank2, colors_rank3 = utils.get_color(X1, X2, X3)


# compute Low projections for each stage using PCA fit on that stage
Low_delay, _, _ = utils.pca_single(hiddens_delay)
Low_res1, _, _ = utils.pca_single(hiddens_res1)
Low_res2, _, _ = utils.pca_single(hiddens_res2)
Low_res3, _, _ = utils.pca_single(hiddens_res3)

lows = [Low_delay, Low_res1, Low_res2, Low_res3]
row_titles = ["delay", "res1", "res2", "res3"]

# prepare global limits - all subplots share the same scale

# define mapping rank -> PC pair: rank 1 -> (PC1,PC2), rank2 -> (PC3,PC4), rank3 -> (PC5,PC6)
rank_to_pair = {1: (0, 1), 2: (2, 3), 3: (4, 5)}

# collect all x and y values across all PCs used in plots
all_xs = []
all_ys = []
for rank in [1, 2, 3]:
    a, b = rank_to_pair[rank]
    all_xs.extend([L[:, a] for L in lows])
    all_ys.extend([L[:, b] for L in lows])

all_xs = np.concatenate(all_xs)
all_ys = np.concatenate(all_ys)

# compute global limits
xmin, xmax = float(all_xs.min()), float(all_xs.max())
ymin, ymax = float(all_ys.min()), float(all_ys.max())
dx = xmax - xmin
dy = ymax - ymin
pad_x = dx * 0.1 if dx != 0 else 0.1
pad_y = dy * 0.1 if dy != 0 else 0.1

# use the same limits for all subplots
global_xlim = (xmin - pad_x, xmax + pad_x)
global_ylim = (ymin - pad_y, ymax + pad_y)

# rows: specify the rank order for each row
orders = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]

# color mapping: allow per-stage (4 rows) and per-PC-pair (3 cols) custom mappings
# colors_matrix is a 4x3 list where each entry is a list of colors for the trials
# default: use the same mapping (colors_rank1, colors_rank2, colors_rank3) for each stage
colors_matrix = [
    [colors_rank1, colors_rank2, colors_rank3],
    [colors_rank1, colors_rank2, colors_rank3],
    [colors_rank2, colors_rank3, colors_rank1],
    [colors_rank3, colors_rank2, colors_rank1],
]

# make 4x3 grid
fig, axes = plt.subplots(4, 3, figsize=(15, 18), sharex=False, sharey=False)
marker = "o"
ms = 40
alpha = 0.9
for r in range(4):
    L = lows[r]
    order = orders[r]
    for c in range(3):
        rank = order[c]
        a, b = rank_to_pair[rank]
        ax = axes[r, c]
        # pick the colors assigned for this stage (r) and this column (c)
        colors = colors_matrix[r][c]
        ax.scatter(L[:, a], L[:, b], c=colors, marker=marker, s=ms, alpha=alpha)
        # use global limits for all subplots
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        # label axes
        if r == 3:
            ax.set_xlabel(f"PC{a+1}")
        if c == 0:
            ax.text(
                -0.08,
                0.5,
                row_titles[r],
                transform=ax.transAxes,
                va="center",
                ha="right",
                fontsize=12,
            )
        # title shows which PC pair is plotted
        ax.set_title(f"PC{a+1} vs PC{b+1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", width=2, direction="in", length=4)

# common suptitle and legend for items
fig.suptitle(
    "Low (stage-wise) PCA projections: rows=stages, cols=PC pairs", fontsize=16
)
legend_handles = [
    plt.Line2D(
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
plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
plt.savefig(
    FIG_DIR / f"{TASK}_PCA_projections_grid.png", bbox_inches="tight", pad_inches=0.12
)
print(f"Saved figure to {FIG_DIR / f'{TASK}_PCA_projections_grid.png'}")

# explained variance bar (from pca fitted on delay_mean earlier)
plt.figure()
label = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]
explained_variance_ratio = pca_x.explained_variance_ratio_[:8]
plt.bar(label, explained_variance_ratio)
plt.savefig(FIG_DIR / f"{TASK}_explained_variance_of_8_PCs.png")
