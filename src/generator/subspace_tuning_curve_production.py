"""
绘制neuron的tuning curve并寻找符合特定条件的neuron。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

import generator.utils as utils
from sklearn.linear_model import Lasso


def set_plot():
    plt.style.use("ggplot")

    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["lines.markeredgewidth"] = 0.003
    plt.rcParams["lines.markersize"] = 3
    plt.rcParams["font.size"] = 14  # 9
    plt.rcParams["legend.fontsize"] = 11  # 7.
    plt.rcParams["axes.facecolor"] = "1"
    plt.rcParams["axes.edgecolor"] = "0"
    plt.rcParams["axes.linewidth"] = "0.7"
    plt.rcParams["axes.grid"] = False

    plt.rcParams["axes.titlesize"] = 18

    plt.rcParams["axes.labelcolor"] = "0"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["xtick.color"] = "0"
    plt.rcParams["ytick.color"] = "0"
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2

    plt.rcParams["font.sans-serif"] = "Arial"


set_plot()

BASE_FONT_SIZE = 12

#### parameters
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name
params = utils.initialize_analysis_legacy(TASK, N_SAMPLES, FILENAME)

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

hiddens = outputs_dict["hiddens"]
hiddens = np.tanh(hiddens)

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"] - 10, :]


base_cmap = plt.get_cmap("tab10")
base_colors = [base_cmap(i) for i in range(6)]

rank_factors = {1: 1.05}
shaded_colors = [utils.shade_color(base_colors[i], rank_factors[1]) for i in range(6)]

ranked_colors = [shaded_colors[0], shaded_colors[3], shaded_colors[1]]

# fmt: off
# overlapping analysis, top_k = 50
common_list = [105, 175, 269, 13, 368, 343, 146, 302, 364, 31, 22, 334, 67, 389, 14, 99, 7, 207, 111, 253, 60, 140, 144, 117, 388, 1, 16, 147, 58, 358, 162, 195, 252, 396, 19, 315, 71, 366, 277, 311, 4, 296, 84, 201, 169, 230, 383, 300, 64, 318]

# overlapping analysis, top_k = 20
# common_list = [105, 175, 269, 13, 368, 343, 146, 302, 364, 31, 22, 334, 67, 389, 14, 99, 7, 207, 111, 253]
# fmt: on

# 构建线性回归模型
# one-hot vector
X = np.zeros([N_SAMPLES, 18])
X1 = []
X2 = []
X3 = []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = PERMS[i]
    X[i, x1] = 1
    X[i, x2 + 6] = 1
    X[i, x3 + 12] = 1
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)


# -----------------------------------
# computation functions
# -----------------------------------
def compute_tuning_curves(hiddens_delay, rank=0, n_item=6):
    """Compute mean delay response for each item.

    hiddens_delay: (N_trials, T_delay, N_HID)
    perms: list/array of tuples (item1,item2,item3) per class
    representative: which item in the tuple to use as the trial's identity (0/1/2)
    returns: tuning_curves (N_HID, n_item), trial_item_ids (N_trials,)
    """

    # per-trial mean over delay
    delay_mean = np.mean(hiddens_delay, axis=1)
    delay_mean = delay_mean[:, np.newaxis, :]

    # coefficient:beta
    model = Lasso(alpha=0.001)  # alpha 是正则化参数
    beta = np.zeros([N_HID, 18])
    for i in np.arange(0, N_HID):
        model.fit(X, delay_mean[:, :, i])
        beta[i, :] = model.coef_

    # compute tuning curves
    tuning_curves = np.zeros([N_HID, n_item])

    beta_rank = beta[:, rank * 6 : (rank + 1) * 6]

    for neuron in range(N_HID):
        tuning_curves[neuron, :] = beta_rank[neuron, :]

    return tuning_curves


def compute_rank_tuning(hiddens_delay, n_item=6):
    """Compute tuning curves for three ranks (representative 0,1,2).

    Returns array of shape (3, N_HID, n_item)
    """
    rank_tunings = []
    for rank in range(3):
        tc = compute_tuning_curves(hiddens_delay, rank=rank, n_item=n_item)
        rank_tunings.append(tc)
    return np.stack(rank_tunings, axis=0)


def fit_cosine_to_curve(curve):
    """Fit cosine forcing phi to be one of the discrete item directions.

    New model (discrete phi): y = A * cos(theta_l - phi_k) + C,
    where theta_l = 2*pi*l/n (l=0..n-1) are item directions and phi_k is chosen
    from the discrete set {theta_l}. For each candidate phi_k we solve linear
    regression for [A, C] (design matrix [cos(theta - phi_k), 1]) and pick the
    candidate with smallest MSE.

    Returns dict with keys: 'A','C','phi' (radians, chosen candidate), 'phi_idx' (0..n-1),
    'pred','mse'. For backward compatibility this does NOT return B/D (set to 0).
    """
    curve = np.asarray(curve, dtype=float)
    n = curve.size
    if n == 0:
        return {
            "B": 0.0,
            "D": 0.0,
            "C": 0.0,
            "A": 0.0,
            "phi": 0.0,
            "phi_idx": 0,
            "pred": curve,
            "mse": 0.0,
        }

    # item directions
    x = np.arange(n)
    thetas = 2 * np.pi * x / n

    best_mse = float("inf")
    best_res = None

    # for each discrete candidate phi (choose phi equal to one of the item directions)
    for k, phi_k in enumerate(thetas):
        # basis vector for this phi: cos(theta - phi_k)
        b = np.cos(thetas - phi_k)
        A_mat = np.vstack([b, np.ones(n)]).T  # design for [A, C]
        # solve least squares for [A, C]
        coeffs, *_ = np.linalg.lstsq(A_mat, curve, rcond=None)
        A_coef, C_coef = coeffs
        pred = A_mat.dot(coeffs)
        mse = float(np.mean((curve - pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_res = {
                "B": 0.0,
                "D": 0.0,
                "C": float(C_coef),
                "A": float(A_coef),
                "phi": float(phi_k),
                "phi_idx": int(k),
                "pred": pred,
                "mse": mse,
            }

    return best_res


def fit_cosine_all(rank_tunings):
    """Fit cosine to every neuron for each rank.

    rank_tunings: (n_ranks, N_hid, N_item)

    Returns a dict with:
        - 'mse': array shape (n_ranks, N_hid) per-neuron mse
        - 'mean': array shape (n_ranks,) mean mse per rank
        - 'std': array shape (n_ranks,) std of mse per rank
        - 'params': nested list of fit dicts per rank and neuron
    """
    rank_tunings = np.asarray(rank_tunings)
    n_ranks, n_hid, n_item = rank_tunings.shape
    mses = np.zeros((n_ranks, n_hid), dtype=float)
    params = [[None for _ in range(n_hid)] for _ in range(n_ranks)]
    # also collect preferred-phase (phi) per rank/neuron and preferred item index
    phis = np.zeros((n_ranks, n_hid), dtype=float)
    phis_deg = np.zeros((n_ranks, n_hid), dtype=float)
    pref_items = np.zeros((n_ranks, n_hid), dtype=int)
    for r in range(n_ranks):
        for i in range(n_hid):
            res = fit_cosine_to_curve(rank_tunings[r, i])
            mses[r, i] = res["mse"]
            params[r][i] = res
            # normalize phi to [0, 2*pi)
            phi = float(res.get("phi", 0.0))
            phi_mod = np.mod(phi, 2 * np.pi)
            phis[r, i] = phi_mod
            phis_deg[r, i] = np.degrees(phi_mod)
            # determine preferred item index (items are at theta_l = 2*pi*l/n_item)
            # the cosine peaks at theta = phi_mod, so map phi to nearest discrete item
            pref = int(np.mod(int(np.round(phi_mod * n_item / (2 * np.pi))), n_item))
            pref_items[r, i] = pref

    mean = mses.mean(axis=1)
    std = mses.std(axis=1)
    return {
        "mse": mses,
        "mean": mean,
        "std": std,
        "params": params,
        "phi": phis,
        "phi_deg": phis_deg,
        "pref_item": pref_items,
    }


# -----------------------------------
# plot functions
# -----------------------------------
def plot_neuron_rank_tuning(
    rank_tunings,
    neuron_idx,
    colors=ranked_colors,
    fig_name=None,
    phi_by_rank=None,
    production=False,
):
    """Plot tuning curves for a single neuron across three ranks on one figure.

    rank_tunings: (3, N_HID, N_item)
    """
    n_item = rank_tunings.shape[2]
    plt.figure(figsize=(3, 2))
    x = np.arange(1, n_item + 1)
    labels = ["Rank1", "Rank2", "Rank3"]
    for r in range(3):
        plt.plot(
            x,
            rank_tunings[r, neuron_idx],
            marker="o",
            markersize=4,
            color=colors[r],
            label=labels[r],
        )
        # if phi_by_rank provided, draw vertical line at preferred direction for this rank
        if phi_by_rank is not None:
            try:
                phi_val = phi_by_rank[r]
            except Exception:
                phi_val = None
            if phi_val is not None and not (
                isinstance(phi_val, float) and np.isnan(phi_val)
            ):
                # detect units: if values exceed 2*pi assume degrees
                if abs(phi_val) > 2 * np.pi:
                    # degrees -> radians
                    phi_rad = np.deg2rad(phi_val)
                else:
                    phi_rad = float(phi_val)
                # normalize to [0, 2*pi)
                phi_rad = np.mod(phi_rad, 2 * np.pi)
                # map phi (radians) to x-position: items at x = 1..n_item correspond to theta = 2*pi*(x-1)/n_item
                x_phi = (phi_rad * n_item) / (2 * np.pi) + 1
                # wrap into [1, n_item]
                if x_phi > n_item:
                    x_phi = x_phi - n_item
                plt.axvline(
                    x=x_phi, color=colors[r], linestyle="--", linewidth=1.5, alpha=0.8
                )

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.spines["bottom"].set_bounds(1, 6)

    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    plt.xticks(x)
    plt.xlabel("Item")
    plt.ylabel("Mean Response")

    if not production:
        plt.title(f"Neuron {neuron_idx} tuning by rank (delay)")
        plt.legend(fontsize=9)

    if fig_name is None:
        fig_name = f"{TASK}_neuron_{neuron_idx}_tuning_by_rank.png"
    out = FIG_DIR / fig_name
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved neuron rank tuning plot to: {out}")
    return out


def plot_rank_legend_only(colors=ranked_colors, fig_name=None):
    """Generate a standalone legend figure for three ranks.

    Useful for inserting into papers as a separate legend.

    colors: list of 3 colors for Rank1, Rank2, Rank3
    fig_name: output filename (default: forward_rank_legend.png)

    Returns path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(2, 3))
    labels = ["Rank1", "Rank2", "Rank3"]

    # Create dummy line objects for legend
    lines = [
        plt.Line2D([0], [0], color=colors[r], linewidth=4, label=labels[r])
        for r in range(3)
    ]

    # Create legend without axis
    ax.legend(
        handles=lines,
        loc="center",
        ncol=1,
        fontsize=11,
        frameon=True,
        edgecolor="gray",
        fancybox=True,
        borderpad=0.8,
        framealpha=0.80,
    )

    # Hide axis
    ax.axis("off")

    if fig_name is None:
        fig_name = f"{TASK}_rank_legend.png"
    out = FIG_DIR / fig_name
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved rank legend to: {out}")
    return out


def plot_combined_figure(
    rank_tunings,
    cos_stats,
    neuron_idx1=15,
    neuron_idx2=84,
    colors=ranked_colors,
    fig_name=None,
    bin_width_deg=30,
    neuron_list=None,
):
    """Create a combined 2-row figure for ICML paper.

    Row 1: Two neuron tuning curves (neuron_idx1, neuron_idx2) + legend
    Row 2: Three phi difference histograms

    Layout optimized for single-column format.
    """
    fig = plt.figure(figsize=(8, 5))

    # GridSpec: 2 rows, 6 columns
    # Row 1: col 0-3 (neuron1), col 3-6 (neuron2)
    # Row 2: col 0-2, col 2-4, col 4-6 (three histograms with minimal overlap)
    gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=0.5, height_ratios=[1, 1])

    # ========== Row 1: Tuning curves ==========
    labels = ["Rank1", "Rank2", "Rank3"]
    x = np.arange(1, 7)

    # Left subplot (neuron_idx1)
    ax1 = fig.add_subplot(gs[0, 0:3])
    for r in range(3):
        ax1.plot(
            x,
            rank_tunings[r, neuron_idx1],
            marker="o",
            markersize=6,
            color=colors[r],
            label=labels[r],
            linewidth=2,
        )
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_bounds(1, 6)
    ax1.spines["left"].set_position(("outward", 4))
    ax1.spines["bottom"].set_position(("outward", 3))
    ax1.set_xticks(x)
    ax1.set_xlabel("Item", fontsize=BASE_FONT_SIZE)
    ax1.set_yticks([-0.2, 0, 0.5], labels=["-0.2", "0", "0.5"])
    ax1.set_ylabel("Coefficient", fontsize=BASE_FONT_SIZE, labelpad=-6)
    ax1.tick_params(labelsize=10)

    # Right subplot (neuron_idx2)
    ax2 = fig.add_subplot(gs[0, 3:6])
    for r in range(3):
        ax2.plot(
            x,
            rank_tunings[r, neuron_idx2],
            marker="o",
            markersize=6,
            color=colors[r],
            label=labels[r],
            linewidth=2,
        )
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_bounds(1, 6)
    ax2.spines["left"].set_position(("outward", 4))
    ax2.spines["bottom"].set_position(("outward", 3))
    ax2.set_xticks(x)
    ax2.set_xlabel("Item", fontsize=BASE_FONT_SIZE)
    ax2.set_yticks([-0.1, 0, 0.2], labels=["-0.1", "0", "0.2"])
    # ax2.set_ylabel("Coefficient", fontsize=8)
    ax2.tick_params(labelsize=10)

    # Add legend to right subplot
    legend_handles = [
        plt.Line2D([0], [0], color=colors[r], linewidth=4, label=labels[r])
        for r in range(3)
    ]
    ax1.legend(
        handles=legend_handles,
        fontsize=9,
        framealpha=0.80,
        loc="upper right",
        frameon=True,
        edgecolor="gray",
        fancybox=True,
        borderpad=0.8,
    )

    # ========== Row 2: Phi difference histograms ==========
    # Compute phi differences
    phis = np.asarray(cos_stats["phi"])
    n_ranks, n_neurons = phis.shape

    # Filter neurons if neuron_list is provided
    if neuron_list is not None:
        phis = phis[:, neuron_list]
        n_neurons = len(neuron_list)

    def angular_diff(a, b):
        d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
        return d

    d01 = np.full(n_neurons, np.nan, dtype=float)
    d12 = np.full(n_neurons, np.nan, dtype=float)
    d20 = np.full(n_neurons, np.nan, dtype=float)
    for i in range(n_neurons):
        d01[i] = angular_diff(phis[0, i], phis[1, i])
        d12[i] = angular_diff(phis[1, i], phis[2, i])
        d20[i] = angular_diff(phis[2, i], phis[0, i])

    d01_deg = np.degrees(d01[~np.isnan(d01)])
    d12_deg = np.degrees(d12[~np.isnan(d12)])
    d20_deg = np.degrees(d20[~np.isnan(d20)])

    max_angle = 180
    bins = np.arange(0, max_angle + bin_width_deg, bin_width_deg)

    # Compute bin counts for y-ticks
    counts_list = [np.histogram(d, bins=bins)[0] for d in (d01_deg, d12_deg, d20_deg)]
    max_count = int(np.max([c.max() if c.size > 0 else 0 for c in counts_list]))
    y_ticks = np.arange(0, max_count + 2, 2)

    # Histogram subplots - share y-axis
    ax_hist1 = fig.add_subplot(gs[1, 0:2])
    ax_hist2 = fig.add_subplot(gs[1, 2:4], sharey=ax_hist1)
    ax_hist3 = fig.add_subplot(gs[1, 4:6], sharey=ax_hist1)

    pair_labels = [
        r"$|\varphi_1-\varphi_2|$",
        r"$|\varphi_2-\varphi_3|$",
        r"$|\varphi_3-\varphi_1|$",
    ]
    colors_gradient = ["#2E86AB", "#A23B72", "#F18F01"]

    axes_hist = [ax_hist1, ax_hist2, ax_hist3]
    data_hist = [d01_deg, d12_deg, d20_deg]

    for ax, data, col in zip(axes_hist, data_hist, colors_gradient):
        n, _, _ = ax.hist(data, bins=bins, edgecolor="white", linewidth=1.5, color=col)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(labelsize=10)

    # Add combined legend at top of the figure
    legend_handles = [
        plt.Line2D([0], [0], color=col, linewidth=4, label=label)
        for col, label in zip(colors_gradient, pair_labels)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.48),
        ncol=3,
        fontsize=9,
        framealpha=0.8,
        columnspacing=1.5,
        borderpad=0.6,
    )

    for idx, ax_hist in enumerate(axes_hist):
        ax_hist.set_xticks(bins)
        ax_hist.set_xlabel(r"$|\Delta\varphi|$ (deg)", fontsize=BASE_FONT_SIZE)
        ax_hist.set_xlim(-5, max_angle + 5)

        if idx != 0:
            ax_hist.spines["left"].set_visible(False)
            ax_hist.tick_params(labelleft=False, left=False)
        else:
            ax_hist.set_ylabel("Count", fontsize=BASE_FONT_SIZE, labelpad=2)
            ax_hist.set_yticks(y_ticks)

    # Add row labels
    fig.text(0.06, 0.86, "A", fontsize=14)
    fig.text(0.06, 0.48, "B", fontsize=14)

    if fig_name is None:
        fig_name = f"{TASK}_combined_figure.pdf"

    out = FIG_DIR / fig_name
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved combined figure to: {out}")
    return out


def compute_nss(cos_stats, neuron_list=None):
    """
    Compute Normalized Strength Score (NSS) for neurons.

    NSS = (Ai - Aj) / (Ai + Aj) where i < j

    Parameters:
        cos_stats: output from fit_cosine_all, contains 'params' with amplitude A
        neuron_list: list of neuron indices to compute NSS for. If None, compute for all neurons.

    Returns:
        nss_dict: dict with keys 'n12', 'n23', 'n13', each containing NSS values for specified neurons
        nss_arrays: dict with raw arrays for all neurons (for debugging/further analysis)
    """
    params = cos_stats["params"]  # nested list: [rank][neuron] -> dict with 'A'

    n_ranks = len(params)  # should be 3
    n_hid = len(params[0])

    # Extract amplitudes A for all neurons
    amplitudes = np.zeros((n_ranks, n_hid))
    for r in range(n_ranks):
        for i in range(n_hid):
            amplitudes[r, i] = np.abs(params[r][i]["A"])

    # Compute NSS for all neurons
    # NSS = (Ai - Aj) / (Ai + Aj), where i < j
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    n12_all = (amplitudes[0] - amplitudes[1]) / (
        amplitudes[0] + amplitudes[1] + epsilon
    )
    n23_all = (amplitudes[1] - amplitudes[2]) / (
        amplitudes[1] + amplitudes[2] + epsilon
    )
    n13_all = (amplitudes[0] - amplitudes[2]) / (
        amplitudes[0] + amplitudes[2] + epsilon
    )

    # Filter to neuron_list if provided
    if neuron_list is not None:
        n12 = n12_all[neuron_list]
        n23 = n23_all[neuron_list]
        n13 = n13_all[neuron_list]
    else:
        n12 = n12_all
        n23 = n23_all
        n13 = n13_all

    nss_dict = {"n12": n12, "n23": n23, "n13": n13}

    nss_arrays = {"n12_all": n12_all, "n23_all": n23_all, "n13_all": n13_all}

    return nss_dict, nss_arrays


def plot_nss_distribution(cos_stats, neuron_list, fig_name=None, bin_width=0.1):
    """
    Plot NSS distribution for common_list neurons across three rank pairs.

    Creates a 1x3 subplot showing:
    - NSS between Rank 1 and Rank 2
    - NSS between Rank 2 and Rank 3
    - NSS between Rank 1 and Rank 3

    Parameters:
        cos_stats: output from fit_cosine_all
        neuron_list: list of neuron indices to plot
        fig_name: output filename (default: forward_nss_distribution.pdf)
        bin_width: bin width for histogram
    """
    # Compute NSS
    nss_dict, _ = compute_nss(cos_stats, neuron_list=neuron_list)

    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # NSS data and labels
    nss_data = [nss_dict["n12"], nss_dict["n23"], nss_dict["n13"]]
    labels = [r"NSS$_{1-2}$", r"NSS$_{2-3}$", r"NSS$_{1-3}$"]

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    # Compute bin range
    all_nss = np.concatenate(nss_data)
    min_val = np.min(all_nss)
    max_val = np.max(all_nss)
    bins = np.arange(min_val - bin_width, max_val + 2 * bin_width, bin_width)

    # Plot histograms
    for idx, (ax, data, label, col) in enumerate(zip(axes, nss_data, labels, colors)):
        ax.hist(data, bins=bins, edgecolor="white", linewidth=1.0, color=col)

        # Format subplot
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, 100)

        base_label_size = 24

        # Compute percentage of neurons with |NSS| > 0.4
        n_total = len(data)
        n_above_threshold = np.sum(np.abs(data) > 0.4)
        percentage = (n_above_threshold / n_total) * 100

        # Add text annotation
        ax.text(
            0.95,
            0.95,
            f"|NSS| > 0.4: {percentage:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=base_label_size - 4,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#333333", alpha=0.8),
        )

        ax.set_xlabel(label, fontsize=base_label_size, labelpad=5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # 设置坐标轴宽度和 tick 宽度
        ax.spines["left"].set_linewidth(2)    # 左 y 轴宽度
        ax.spines["bottom"].set_linewidth(2)  # x 轴宽度
        ax.tick_params(labelsize=base_label_size - 2, width=2, length=8)  # tick 宽度和长度

        # 只在最左侧显示y轴
        if idx == 0:
            ax.set_ylabel("Count", fontsize=base_label_size)
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(labelleft=False, left=False)

    plt.tight_layout()

    if fig_name is None:
        fig_name = f"{TASK}_nss_distribution.pdf"
    out = FIG_DIR / fig_name
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved NSS distribution plot to: {out}")
    return out


if __name__ == "__main__":
    # Compute full rank_tunings for all neurons
    full_rank_tunings = compute_rank_tuning(hiddens_delay, n_item=6)
    full_cos_stats = fit_cosine_all(full_rank_tunings)

    for r in range(full_cos_stats["mse"].shape[0]):
        print(
            f"Rank {r}: MSE mean = {full_cos_stats['mean'][r]:.6f}, std = {full_cos_stats['std'][r]:.6f}"
        )

    # Generate combined figure: use absolute indices for tuning curves, common_list for histograms
    plot_combined_figure(
        full_rank_tunings,
        full_cos_stats,
        neuron_idx1=86,  # gain-modulation
        neuron_idx2=370,  # preference shift
        colors=[shaded_colors[0], shaded_colors[3], shaded_colors[1]],
        bin_width_deg=30,
        neuron_list=common_list,
        fig_name="Figure6_revised.pdf",
    )

    # Plot NSS distribution for common_list neurons
    plot_nss_distribution(
        full_cos_stats,
        # neuron_list=common_list,
        neuron_list=range(N_HID),
        fig_name="sfig_nss_distribution.pdf",
        bin_width=0.1,
    )

    # for i in range(N_HID):
    #     plot_neuron_rank_tuning(
    #         full_rank_tunings,
    #         neuron_idx=i,
    #         colors=[shaded_colors[0], shaded_colors[3], shaded_colors[1]],
    #         fig_name=f"{TASK}_neuron_{i}_tuning_by_rank.png",
    #     )
