"""
绘制neuron的tuning curve并寻找符合特定条件的neuron。
使用实验当中的GLM方法。

实验分析当中使用了late delay period的神经活动数据。
We focused on neural activity during the late delay period
(1 s before the “fixationoff” go signal) while the monkeys
maintained length-2 or -3 spatial sequences in memory.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from scipy.optimize import minimize

import generator.utils as utils


# ============================================================
#  Parameters & data loading
# ============================================================
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

hiddens = np.tanh(outputs_dict["hiddens"])
hiddens_delay = hiddens[:, tps["delay_end"] - 10 : tps["delay_end"], :]

base_cmap = plt.get_cmap("tab10")
base_colors = [base_cmap(i) for i in range(6)]


def compute_one(class_i):
    class_i = class_i % N_CLASS
    item_1, item_2, item_3 = PERMS[class_i]
    return item_1, item_2, item_3


# ============================================================
#  Step 1. Build design matrix (18 one-hot task variables)
# ============================================================
X = np.zeros([N_SAMPLES, 18])
for i in np.arange(N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X[i, x1] = 1
    X[i, x2 + 6] = 1
    X[i, x3 + 12] = 1

delay_mean = np.mean(hiddens_delay, axis=1)
delay_mean = delay_mean[:, np.newaxis, :]

# ============================================================
#  Step 2. Lasso regression  → β_i(r,l)
# ============================================================
model = Lasso(alpha=0.001)
beta = np.zeros([N_HID, 18])
for i in range(N_HID):
    model.fit(X, delay_mean[:, :, i])
    beta[i, :] = model.coef_


# ============================================================
#  Step 3. PCA per rank, and anchor rotation + sign check
# ============================================================
def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


rank_scores, rank_coeffs = [], []
anchor_item = 2  # MATLAB 第3项（Python索引2）

for r in range(3):
    Br = beta[:, r * 6 : (r + 1) * 6]
    mean_r = Br.mean(axis=1, keepdims=True)
    Br_dm = Br - mean_r

    pca = PCA(n_components=2)
    # fit PCA on items (6 samples) -> components_ becomes available
    pca.fit(Br_dm.T)
    coeff = pca.components_.T  # (N,2)
    score = pca.transform(Br_dm.T)  # (6,2)

    # --- anchor rotation ---
    x, y = score[anchor_item, :]
    ra = -np.arctan2(y, x)
    R = rotation(ra)
    score = (R @ score.T).T
    coeff = coeff @ np.linalg.inv(R)

    # --- flip check ---
    check = [anchor_item - 1, anchor_item + 1]
    if check[0] >= 0 and check[1] < 6:
        st = np.sign(score[check, 1])
        if st[0] > st[1]:
            score = np.fliplr(score)
            coeff = np.fliplr(coeff)

    rank_scores.append(score)
    rank_coeffs.append(coeff)

# ============================================================
#  Step 4. fmincon-equivalent optimization (θ_r, σ_r, f_l)
# ============================================================
# initialize parameters
theta0 = [np.pi / 2, np.pi / 2, np.pi / 2]
sigma0 = [0.8, 0.8, 1.0]
template = np.mean([rank_scores[0].T, rank_scores[1].T], axis=0)
x0 = np.concatenate([theta0, sigma0, template.flatten()])


def Rscale(theta, sigma):
    return np.array(
        [
            [sigma * np.cos(theta), -sigma * np.sin(theta)],
            [sigma * np.sin(theta), sigma * np.cos(theta)],
        ]
    )


def objective(x):
    theta = x[0:3]
    sigma = x[3:6]
    f = x[6:].reshape(2, 6)
    loss = 0.0
    for r in range(3):
        R = Rscale(theta[r], sigma[r])
        loss += np.linalg.norm(R @ rank_scores[r].T - f, "fro")
    norm_term = sum(
        np.linalg.norm(Rscale(theta[r], sigma[r]) @ rank_scores[r].T, "fro")
        for r in range(3)
    )
    return loss / (norm_term + 1e-9)


res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 10000})
para = res.x
theta, sigma = para[0:3], para[3:6]
filler = para[6:].reshape(2, 6)

print("[VAR] filler:", filler)


# ============================================================
#  Step 5. Compute similarity and Q_r, λ_r
# ============================================================
def inv_Rscale(theta, sigma):
    c, s = np.cos(theta), np.sin(theta)
    return (1.0 / sigma) * np.array([[c, s], [-s, c]])


Sim_score = []
for r in range(3):
    invR = inv_Rscale(theta[r], sigma[r])
    loss = np.linalg.norm(invR @ filler - rank_scores[r].T, "fro") / np.linalg.norm(
        rank_scores[r], "fro"
    )
    Sim_score.append(1 - loss**2)

Q_list = []
for r in range(3):
    R = rotation(theta[r])
    Qr = rank_coeffs[r] @ np.linalg.inv(R)
    Q_list.append(Qr)

# gain factors normalization
lam_raw = 1.0 / np.array(sigma)
scaler = lam_raw[0]
lambda_list = lam_raw / scaler
f_norm = filler * scaler

# ============================================================
#  Step 6. Plot hexagons
# ============================================================
labels = [1, 2, 3, 4, 5, 6]
fig, axes = plt.subplots(1, 3, figsize=(21, 8))

all_x = []
all_y = []

for ridx in range(3):
    ax = axes[ridx]
    R = rotation(theta[ridx])
    coords = (R @ rank_scores[ridx].T).T

    for i in range(6):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            s=100,
            color=base_colors[i],
            label=str(labels[i]),
        )
        if i > 0:
            ax.plot(
                [coords[i - 1, 0], coords[i, 0]],
                [coords[i - 1, 1], coords[i, 1]],
                color="gray",
            )

    ax.plot([coords[-1, 0], coords[0, 0]], [coords[-1, 1], coords[0, 1]], color="gray")

    ax.set_title(f"Rank {ridx+1} subspace (λ={lambda_list[ridx]:.2f})")
    ax.set_xlabel("rPC1")
    if ridx == 0:
        ax.set_ylabel("rPC2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)


handles, labels_legend = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels_legend, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.98)
)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(
    FIG_DIR / f"{TASK}_gain_modulation_hexagon_precise.png",
    bbox_inches="tight",
    pad_inches=0.12,
)
print(
    f"[PLOT] Saved precise gain-modulation plot to {FIG_DIR}/{TASK}_gain_modulation_hexagon_precise.png"
)

# ============================================================
#  Step 7. Use PCA as a comparison
# ============================================================

## pca analysis
# - the first dimension of h_pca is item idx, thus 6
# - the second dimension is n_components_ev, yet with the cumulative variance result,
#   only the first two PCs are useful and retained
# - the third dimension is rank idx (1,2,3)
n_components_ev = 6
h_pca = np.zeros([6, n_components_ev, 3])


for i in [1, 2, 3]:
    analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
    pca_EV = PCA(n_components=n_components_ev)
    pca_EV.fit(analy)
    h_pca[:, :, i - 1] = pca_EV.transform(analy)

label = [1, 2, 3, 4, 5, 6]

# Combine the three subspace PC1-vs-PC2 plots into one figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# compute global x/y limits across all subspaces (PC1 vs PC2)
all_x = h_pca[:, 0, :].ravel()
all_y = h_pca[:, 1, :].ravel()
xmin, xmax = float(all_x.min()), float(all_x.max())
ymin, ymax = float(all_y.min()), float(all_y.max())
# add padding (10%)
dx = xmax - xmin
dy = ymax - ymin
pad_x = dx * 0.1 if dx != 0 else 0.1
pad_y = dy * 0.1 if dy != 0 else 0.1
xmin -= pad_x
xmax += pad_x
ymin -= pad_y
ymax += pad_y

for idx, j in enumerate([0, 1, 2]):
    ax = axes[idx]
    last_point = None
    for i in np.arange(n_components_ev):
        ax.scatter(h_pca[i, 0, j], h_pca[i, 1, j], label=str(label[i]), s=70)
        if last_point is not None:
            ax.plot(
                [last_point[0], h_pca[i, 0, j]],
                [last_point[1], h_pca[i, 1, j]],
                color="gray",
            )
        last_point = (h_pca[i, 0, j], h_pca[i, 1, j])
    if last_point is not None:
        ax.plot(
            [last_point[0], h_pca[0, 0, j]],
            [last_point[1], h_pca[0, 1, j]],
            color="gray",
        )

    ax.set_xlabel("PC1")
    if idx == 0:
        ax.set_ylabel("PC2")
    ax.set_title(f"Subspace {idx+1}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    # enforce identical axis limits and equal aspect ratio across subplots
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        # fallback if backend doesn't support adjustable box
        pass

# put a shared legend above the subplots
handles, labels_legend = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels_legend,
    loc="upper center",
    ncol=len(label),
    bbox_to_anchor=(0.5, 0.98),
)
# reserve more top margin so legend isn't cut; use bbox_inches='tight' when saving to include legend
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(
    FIG_DIR / f"{TASK}_PCA_analysis_hexagon.png", bbox_inches="tight", pad_inches=0.12
)
print(f"[PLOT] Saved PCA hexagon plot to {FIG_DIR}/{TASK}_PCA_analysis_hexagon.png")

raise SystemExit


def compute_tuning_curves_glm(beta, n_item=6):
    """
    Use beta to compute tuning curves of three ranks
    for all ``n_item`` items.
    """

    beta_ranks = [beta[:, 0:6], beta[:, 6:12], beta[:, 12:18]]
    tuning_curves = np.zeros((3, N_HID, n_item))  # (3, N_HID, n_item)

    for r in range(3):
        for item in range(n_item):
            tuning_curves[r, :, item] = beta_ranks[r][:, item]

    return tuning_curves


def plot_neuron_rank_tuning_glm(rank_tunings, neuron_idx, colors=None, fig_name=None):
    """Plot tuning curves for a single neuron across three ranks on one figure.

    rank_tunings: (3, N_HID, N_item)
    """
    if colors is None:
        # use a slightly more yellow hue for rank3 (was 'tab:orange')
        colors = ["#026caf", "#ce5327", "#e3ac27"]  # rank1, rank2, rank3

    n_item = rank_tunings.shape[2]
    plt.figure(figsize=(6, 4))
    x = np.arange(1, n_item + 1)
    labels = ["Rank1", "Rank2", "Rank3"]
    for r in range(3):
        plt.plot(
            x, rank_tunings[r, neuron_idx], marker="o", color=colors[r], label=labels[r]
        )
    plt.xticks(x)
    plt.xlabel("Item")
    plt.ylabel("Coefficient")
    plt.title(f"Neuron {neuron_idx} tuning by rank (late delay)")
    plt.legend()
    if fig_name is None:
        fig_name = f"{TASK}_neuron_{neuron_idx}_tuning_by_rank.png"
    out = FIG_DIR / fig_name
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved neuron rank tuning plot to: {out}")
    return out


def compute_tuning_curves(hiddens, perms, representative=0, n_item=6):
    """Compute mean response per neuron for each item.

    hiddens: (N_trials, T, N_HID)
    perms: list/array of tuples (item1,item2,item3) per class
    representative: which item in the tuple to use as the trial's identity (0/1/2)
    returns: tuning_curves (N_HID, n_item), trial_item_ids (N_trials,)
    """
    N_trials = hiddens.shape[0]
    N_hid = hiddens.shape[2]

    # per-trial mean over delay
    h_mean = hiddens.mean(axis=1)  # (N_trials, N_hid)

    # trial -> item id mapping
    trial_item_ids = np.zeros(N_trials, dtype=int)
    for i in range(N_trials):
        class_i = i % len(perms)
        items = perms[class_i]
        trial_item_ids[i] = int(items[representative])

    tuning_curves = np.zeros((N_hid, n_item))
    for k in range(n_item):
        idx = np.where(trial_item_ids == k)[0]
        if idx.size > 0:
            tuning_curves[:, k] = h_mean[idx].mean(axis=0)
        else:
            tuning_curves[:, k] = 0.0

    return tuning_curves, trial_item_ids


def compute_rank_tuning(hiddens, perms, n_item=6):
    """Compute tuning curves for three ranks (representative 0,1,2).

    Returns array of shape (3, N_HID, n_item)
    """
    rank_tunings = []
    for rep in range(3):
        tc, _ = compute_tuning_curves(hiddens, perms, representative=rep, n_item=n_item)
        rank_tunings.append(tc)
    return np.stack(rank_tunings, axis=0)


def plot_neuron_rank_tuning(hiddens, neuron_idx, colors=None, fig_name=None):
    """Plot tuning curves for a single neuron across three ranks on one figure."""
    rank_tunings = compute_rank_tuning(hiddens, PERMS, n_item=6)
    if colors is None:
        colors = ["tab:blue", "tab:red", "tab:orange"]  # rank1, rank2, rank3

    n_item = rank_tunings.shape[2]
    plt.figure(figsize=(6, 4))
    x = np.arange(1, n_item + 1)
    labels = ["Rank1", "Rank2", "Rank3"]
    for r in range(3):
        plt.plot(
            x, rank_tunings[r, neuron_idx], marker="o", color=colors[r], label=labels[r]
        )
    plt.xticks(x)
    plt.xlabel("Item")
    plt.ylabel("Mean response (delay)")
    plt.title(f"Neuron {neuron_idx} tuning by rank (delay)")
    plt.legend()
    if fig_name is None:
        fig_name = f"{TASK}_neuron_{neuron_idx}_tuning_by_rank.png"
    out = FIG_DIR / fig_name
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved neuron rank tuning plot to: {out}")
    return out


if __name__ == "__main__":
    tuning_curves = compute_tuning_curves_glm(beta, n_item=6)  # (3, N_HID, 6)

    neuron_idx = 10

    plot_neuron_rank_tuning_glm(
        tuning_curves,
        neuron_idx,
        fig_name=f"{TASK}_neuron_{neuron_idx}_tuning_by_rank_delay_glm.png",
    )

    plot_neuron_rank_tuning(
        hiddens_delay,
        neuron_idx,
        fig_name=f"{TASK}_neuron_{neuron_idx}_tuning_by_rank_delay.png",
    )
    plt.show()
