"""
动力学投影分析：
识别在 encoding 阶段对子空间旋转贡献最大的神经元
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import generator.utils as utils

# ============================================================
# parameters
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

# ============================================================
# load hidden states
# shape: (trial, time, neuron)
# ============================================================
hiddens = outputs_dict["hiddens"]
Win = outputs_dict["Win"]
Wout = outputs_dict["Wout"]

hiddens = np.tanh(hiddens)


# ============================================================
# utility: task permutation
# ============================================================
def compute_one(class_i):
    if class_i >= N_CLASS:
        item_1, item_2, item_3 = PERMS[class_i - N_CLASS]
        return item_3, item_2, item_1
    else:
        item_1, item_2, item_3 = PERMS[class_i]
        return item_1, item_2, item_3


# ============================================================
# extract encoding / delay hidden states
# ============================================================
h_enc = hiddens[:, tps["stim1_start"] : tps["stim3_end"], :]
h_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

# trial-averaged state clouds
X_enc = np.mean(h_enc, axis=1)  # (trial, neuron)
X_delay = np.mean(h_delay, axis=1)  # (trial, neuron)


# ============================================================
# compute rank subspace (PCA) for encoding
# ============================================================
def compute_pca_basis(X, k=2):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T  # shape: (N_HID, k)


k = 2
U_enc = compute_pca_basis(X_enc, k=k)

# ============================================================
# compute hidden state velocity dx/dt
# ============================================================
dt = 1.0
dhiddens = np.diff(hiddens, axis=1) / dt  # (trial, time-1, neuron)

dh_enc = dhiddens[:, tps["stim1_start"] : tps["stim1_end"] - 1, :]

# 平均动力学速度（population-level）
V_enc = np.mean(dh_enc, axis=(0, 1))  # shape: (N_HID,)

# ============================================================
# project velocity onto orthogonal complement of rank subspace
# ============================================================
P_perp = np.eye(N_HID) - U_enc @ U_enc.T
V_rot = P_perp @ V_enc  # rotational / tilting component

# ============================================================
# neuron-wise rotation contribution
# ============================================================
rotation_contrib = np.abs(V_rot)
rotation_contrib /= rotation_contrib.sum()

# 排序
idx_sorted = np.argsort(rotation_contrib)[::-1]
top_frac = 0.05
top_k = int(top_frac * N_HID)

top_neurons = idx_sorted[:top_k]

# ============================================================
# visualize contribution distribution
# ============================================================
plt.figure(figsize=(6, 4))
plt.plot(np.sort(rotation_contrib)[::-1])
plt.axvline(top_k, color="r", linestyle="--", label=f"Top {int(top_frac*100)}%")
plt.xlabel("Neuron (sorted)")
plt.ylabel("Rotation contribution")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "rotation_contribution_ranked.png")
plt.close()

# ============================================================
# visualize contribution histogram
# ============================================================
plt.figure(figsize=(6, 4))
plt.hist(rotation_contrib, bins=40)
plt.xlabel("Rotation contribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "rotation_contribution_hist.png")
plt.close()

# ============================================================
# compare activity: top rotation neurons vs others
# ============================================================
mean_act_enc = np.mean(h_enc, axis=(0, 1))

plt.figure(figsize=(6, 4))
plt.boxplot(
    [mean_act_enc[top_neurons], mean_act_enc[idx_sorted[top_k:]]],
    tick_labels=["Top rotation neurons", "Others"],
)
plt.ylabel("Mean activity (encoding)")
plt.tight_layout()
plt.savefig(FIG_DIR / "rotation_neuron_activity_compare.png")
plt.close()

print(f"Top {top_frac*100:.0f}% neurons contributing to rotation: {top_k}")
print("Indices:", top_neurons[:10], "...")


# ============================================================
# plot cross-subspace angles between layers
# ============================================================


def plot_win_angles(Win, hiddens, name_suffix=""):
    def extract_subspace_bases(H):
        """提取H的三个二维子空间基底矩阵（每列为一个主成分）"""
        H = np.mean(H, axis=1).reshape(-1, N_HID)
        pca = PCA(n_components=6).fit(H)
        # 每个子空间取前两列基底
        bases = {
            "sub1": pca.components_[[0, 1]].T,  # PC1-2 (shape: n_neurons × 2)
            "sub2": pca.components_[[2, 3]].T,  # PC3-4
            "sub3": pca.components_[[4, 5]].T,  # PC5-6
        }
        return bases

    hidden_in_1 = hiddens[:, tps["stim1_start"] : tps["stim1_off"], :]
    hidden_in_2 = hiddens[:, tps["stim2_start"] : tps["stim2_off"], :]
    hidden_in_3 = hiddens[:, tps["stim3_start"] : tps["stim3_off"], :]

    # 提取基底
    bases_H1 = extract_subspace_bases(hidden_in_1)
    bases_H3 = extract_subspace_bases(hidden_in_2)
    bases_H5 = extract_subspace_bases(hidden_in_3)

    ang_sub1 = [
        np.mean(utils.neo_compute_principal_angles(bases_H1["sub1"], Win.T)),
        np.mean(utils.neo_compute_principal_angles(bases_H3["sub1"], Win.T)),
        np.mean(utils.neo_compute_principal_angles(bases_H5["sub1"], Win.T)),
    ]

    ang_sub2 = [
        np.mean(utils.neo_compute_principal_angles(bases_H1["sub2"], Win.T)),
        np.mean(utils.neo_compute_principal_angles(bases_H3["sub2"], Win.T)),
        np.mean(utils.neo_compute_principal_angles(bases_H5["sub2"], Win.T)),
    ]

    ang_sub3 = [
        np.mean(utils.neo_compute_principal_angles(bases_H1["sub3"], Win.T)),
        np.mean(utils.neo_compute_principal_angles(bases_H3["sub3"], Win.T)),
        np.mean(utils.neo_compute_principal_angles(bases_H5["sub3"], Win.T)),
    ]

    plt.figure(figsize=(10, 6))

    # 绘图并添加标签
    plt.plot(
        ["In1", "In2", "In3"],
        ang_sub1,
        label="Sub1",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        ["In1", "In2", "In3"],
        ang_sub2,
        label="Sub2",
        marker="s",
        linestyle="--",
        color="green",
    )
    plt.plot(
        ["In1", "In2", "In3"],
        ang_sub3,
        label="Sub3",
        marker="^",
        linestyle="-.",
        color="red",
    )

    plt.title("angle during input stage", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=10)  # 自动选择最佳图例位置
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"input_time_ang_{name_suffix}.png")
    print(f"Saved figure: input_time_ang_{name_suffix}.png")
    plt.close()


def plot_cross_angles(hiddens, name_suffix=""):

    hiddens_input1 = hiddens[:, tps["stim1_start"] : tps["stim1_off"], :]
    hiddens_input2 = hiddens[:, tps["stim2_start"] : tps["stim2_off"], :]
    hiddens_input3 = hiddens[:, tps["stim3_start"] : tps["stim3_off"], :]

    hiddens_input1_delay = hiddens[:, tps["stim1_off"] : tps["stim1_end"], :]
    hiddens_input2_delay = hiddens[:, tps["stim2_off"] : tps["stim2_end"], :]
    hiddens_input3_delay = hiddens[:, tps["stim3_off"] : tps["stim3_end"], :]

    # hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

    INPUT1 = np.mean(hiddens_input1, axis=1).reshape(-1, N_HID)
    INPUT2 = np.mean(hiddens_input2, axis=1).reshape(-1, N_HID)
    INPUT3 = np.mean(hiddens_input3, axis=1).reshape(-1, N_HID)

    INPUT1_DELAY = np.mean(hiddens_input1_delay, axis=1).reshape(-1, N_HID)
    INPUT2_DELAY = np.mean(hiddens_input2_delay, axis=1).reshape(-1, N_HID)
    INPUT3_DELAY = np.mean(hiddens_input3_delay, axis=1).reshape(-1, N_HID)

    # DELAY = np.mean(hiddens_delay, axis=1).reshape(-1, N_HID)

    def plot_cross_angles_sub(angle_matrix, labels):
        """绘制所有子空间间夹角热力图"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            angle_matrix,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Average Angle (Degrees)"},
        )
        plt.title("Cross-Subspace Principal Angles Between Layers")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(FIG_DIR / f"cross_subspace_principal_angles_{name_suffix}.png")
        print(f"Saved figure: cross_subspace_principal_angles_{name_suffix}.png")
        plt.close()

    layers = {
        "Input1": INPUT1,
        "Input1_D": INPUT1_DELAY,
        "Input2": INPUT2,
        "Input2_D": INPUT2_DELAY,
        "Input3": INPUT3,
        "Input3_D": INPUT3_DELAY,
        # "Delay": DELAY,
    }

    subspace_bases, angle_matrix, labels = utils.compute_subspace_angles(layers)
    plot_cross_angles_sub(angle_matrix, labels)


def set_hiddens_subset(hiddens, neuron_indices):
    for idx in neuron_indices:
        hiddens[:, :, idx] = 0.0

    return hiddens


hiddens_top = set_hiddens_subset(hiddens.copy(), idx_sorted[top_k:])
hiddens_others = set_hiddens_subset(hiddens.copy(), idx_sorted[:top_k])

plot_cross_angles(hiddens, name_suffix="all_neurons")
plot_cross_angles(hiddens_top, name_suffix="top_rotation_neurons")
plot_cross_angles(hiddens_others, name_suffix="other_neurons")

plot_win_angles(Win, hiddens, name_suffix="all_neurons")
plot_win_angles(Win, hiddens_top, name_suffix="top_rotation_neurons")
plot_win_angles(Win, hiddens_others, name_suffix="other_neurons")
