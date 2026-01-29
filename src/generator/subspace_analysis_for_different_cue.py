"""
还没搞懂这个代码是干什么的……并且疑似有问题。
!KEY
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

import generator.utils as utils

#### parameters
TASK = "switch"
N_SAMPLES = 240
FILENAME = Path(__file__).name
params = utils.initialize_analysis(TASK, N_SAMPLES, FILENAME)

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

## dataset
hiddens = outputs_dict["hiddens"]
Wout = outputs_dict["Wout"]
Win = outputs_dict["Win"][25:27, :]

hiddens = np.tanh(hiddens)


def compute_one(class_i):
    if class_i >= N_CLASS:
        # backward
        item_1, item_2, item_3 = PERMS[class_i - N_CLASS]
        return item_3, item_2, item_1

    else:
        # forward
        item_1, item_2, item_3 = PERMS[class_i]
        return item_1, item_2, item_3


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

hiddens_pre_cue = hiddens[:, tps["pre_cue_start"] : tps["cue_start"], :]
hiddens_cue_f = hiddens[0:N_CLASS, tps["cue_start"] : tps["cue_end"], :]
hiddens_cue_b = hiddens[N_CLASS:, tps["cue_start"] : tps["cue_end"], :]
hiddens_post_cue = hiddens[:, tps["cue_end"] : tps["post_cue_end"], :]

hidden_out_1_f = hiddens[0:N_CLASS, tps["target1_start"] : tps["target1_end"], :]
hidden_out_2_f = hiddens[0:N_CLASS, tps["target2_start"] : tps["target2_end"], :]
hidden_out_3_f = hiddens[0:N_CLASS, tps["target3_start"] : tps["target3_end"], :]

hidden_out_1_b = hiddens[N_CLASS:, tps["target1_start"] : tps["target1_end"], :]
hidden_out_2_b = hiddens[N_CLASS:, tps["target2_start"] : tps["target2_end"], :]
hidden_out_3_b = hiddens[N_CLASS:, tps["target3_start"] : tps["target3_end"], :]


def plot_cross_angles(angle_matrix, labels):
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
    plt.savefig(FIG_DIR / "corss_subspace_principal_angles_between_layers.png")
    plt.close()


BACKWARD_POST_CUE = np.mean(hiddens_post_cue[0:N_CLASS, :, :], axis=1).reshape(
    -1, N_HID
)
FORWARD_POST_CUE = np.mean(hiddens_post_cue[N_CLASS:, :, :], axis=1).reshape(-1, N_HID)


layers = {"backward-postcue": BACKWARD_POST_CUE, "forward-postcue": FORWARD_POST_CUE}
subspace_bases, angle_matrix, labels = utils.compute_subspace_angles(layers)
plot_cross_angles(angle_matrix, labels)


colors_list = ["blue", "orange", "green", "red", "purple", "brown"]
colors1 = [colors_list[i] for i in X1]
colors2 = [colors_list[i] for i in X2]
colors3 = [colors_list[i] for i in X3]

Low_forward, var_explained_forward, _ = utils.pca_single(
    hiddens_post_cue[0:N_CLASS, :, :]
)
Low_backward, var_explained_backward, _ = utils.pca_single(
    hiddens_post_cue[N_CLASS:, :, :]
)


# 这里开始是计算贡献度的
# 加载隐藏层数据 h_flat (1200 samples, N_HID features)

OUT1_F = hidden_out_1_f.reshape(-1, N_HID)
OUT2_F = hidden_out_2_f.reshape(-1, N_HID)
OUT3_F = hidden_out_3_f.reshape(-1, N_HID)

# 定义全局参数
pc_groups = [(0, 2), (2, 4), (4, 6)]
group_labels = ["Sub1", "Sub2", "Sub3"]


def compute_ratio_contributions(H, Wout):
    """计算单个H的分组贡献比 (仅层内比例)"""
    # 1. 独立PCA提取主成分
    H = np.mean(H, axis=1).reshape(-1, N_HID)
    pca = PCA(n_components=6).fit(H)
    components = pca.components_

    # 2. 计算原始贡献
    contribs_raw = []
    for start, end in pc_groups:
        group_comps = components[start:end]
        # 投影方差
        var_proj = sum((H @ pc).var() for pc in group_comps)
        # Wout投影强度（平方和）
        w_proj_sq = sum(
            ((Wout.T @ pc.reshape(-1, 1)).flatten() ** 2).sum() for pc in group_comps
        )
        contribs_raw.append(var_proj * w_proj_sq)

    # 3. 转换为比例（总和为1）
    total = sum(contribs_raw)
    return [c / total for c in contribs_raw]


# 计算贡献比例
contrib_ratios = {
    "Out1": compute_ratio_contributions(OUT1_F, Wout),
    "Out2": compute_ratio_contributions(OUT2_F, Wout),
    "Out3": compute_ratio_contributions(OUT3_F, Wout),
}

plt.figure(figsize=(14, 4))
for i, h in enumerate(["Out1", "Out2", "Out3"]):
    plt.subplot(1, 3, i + 1)
    plt.bar(group_labels, contrib_ratios[h], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylim(0, 1)
    plt.ylabel("Relative Contribution")
    plt.title(f"{h}  Contributions")
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "forward_Wout_contributions.png")


OUT1_B = hidden_out_1_b.reshape(-1, N_HID)
OUT2_B = hidden_out_2_b.reshape(-1, N_HID)
OUT3_B = hidden_out_3_b.reshape(-1, N_HID)

# 定义全局参数
pc_groups = [(0, 2), (2, 4), (4, 6)]
group_labels = ["Sub1", "Sub2", "Sub3"]


# 计算贡献比例
contrib_ratios = {
    "Out1": compute_ratio_contributions(OUT1_B, Wout),
    "Out2": compute_ratio_contributions(OUT2_B, Wout),
    "Out3": compute_ratio_contributions(OUT3_B, Wout),
}


plt.figure(figsize=(14, 4))
for i, h in enumerate(["Out1", "Out2", "Out3"]):
    plt.subplot(1, 3, i + 1)
    plt.bar(group_labels, contrib_ratios[h], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylim(0, 1)
    plt.ylabel("Relative Contribution")
    plt.title(f"{h}  Contributions")
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "backward_Wout_contributions.png")


# 加载隐藏层数据 h_flat (1200 samples, N_HID features)
x = hiddens_cue_f
h = np.mean(x, axis=1)
H6 = h.reshape(-1, N_HID)

# 定义全局参数
pc_groups = [(0, 2), (2, 4), (4, 6)]
group_labels = ["Subspace1", "Subspace2", "Subspace3"]


# 计算贡献比例
contrib_ratios = {"Cue": compute_ratio_contributions(H6, Win.T)}
# 可视化
plt.figure(figsize=(14, 4))
for i, h in enumerate(["Cue"]):
    plt.subplot(1, 3, i + 1)
    plt.bar(group_labels, contrib_ratios[h], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylim(0, 1)
    plt.ylabel("Relative Contribution")
    plt.title(f"{h}  Contributions")
    plt.grid(alpha=0.3)

    utils.set_plot(7)
    plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "forward_cue_contribution.png")


# 加载隐藏层数据 h_flat (1200 samples, N_HID features)
x = hiddens_cue_b
h = np.mean(x, axis=1)
H6 = h.reshape(-1, N_HID)

# 定义全局参数
pc_groups = [(0, 2), (2, 4), (4, 6)]
group_labels = ["Subspace1", "Subspace2", "Subspace3"]


# 计算贡献比例
contrib_ratios = {"Cue": compute_ratio_contributions(H6, Win.T)}
# 可视化
plt.figure(figsize=(14, 4))
for i, h in enumerate(["Cue"]):
    plt.subplot(1, 3, i + 1)
    plt.bar(group_labels, contrib_ratios[h], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylim(0, 1)
    plt.ylabel("Relative Contribution")
    plt.title(f"{h}  Contributions")
    plt.grid(alpha=0.3)

    utils.set_plot(7)
    plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "backward_cue_contribution.png")
