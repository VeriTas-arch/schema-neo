"""
Summary
-------
- backward 任务
- 对应Geometry of sequence working memory in macaque prefrontal cortex的fig.2A
- 分析Delay阶段的序列信息与内容信息的存储方式。此外也涵盖了fig.S3的部分内容。

Notes
-----
- ppt 已完成
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

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


hiddens = outputs_dict["hiddens"]
batch = outputs_dict["batch"]

hiddens = np.tanh(hiddens)

hiddens_delay = hiddens[:, tps["cue_start"] - 6 : tps["cue_start"] - 2, :]

delay_mean = np.mean(hiddens_delay, axis=1)
pca_x = PCA(n_components=8).fit(delay_mean)
Low = pca_x.transform(delay_mean)


def compute_one(class_i):
    class_i = class_i % N_CLASS
    item_1, item_2, item_3 = PERMS[class_i]
    return item_1, item_2, item_3


# 构建线性回归模型
# one-hot vector
X = np.zeros([N_SAMPLES, 18])
X1 = []
X2 = []
X3 = []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X[i, x1] = 1
    X[i, x2 + 6] = 1
    X[i, x3 + 12] = 1
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)


base_colors, colors_rank1, colors_rank2, colors_rank3 = utils.get_color(X1, X2, X3)

delay_mean = delay_mean[:, np.newaxis, :]  # shape (120, 1, N_HID)

# coefficient:beta
model = Lasso(alpha=0.001)  # alpha 是正则化参数
beta = np.zeros([N_HID, 18])
for i in np.arange(0, N_HID):
    model.fit(X, delay_mean[:, :, i])
    beta[i, :] = model.coef_

## pca analysis
n_components_ev = 6
components = np.zeros([N_HID, 6])
h_pca = np.zeros([6, n_components_ev, 3])

for i in [1, 2, 3]:
    analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
    pca_THETA = PCA(n_components=2)
    pca_EV = PCA(n_components=n_components_ev)
    pca_THETA.fit(analy)
    components[:, (i - 1) * 2 : i * 2] = pca_THETA.components_.T

    pca_EV.fit(analy)
    h_pca[:, :, i - 1] = pca_EV.transform(analy)

### 计算不同子空间夹角
theta = np.zeros([2, 3])
V = np.transpose(components[:, 0:2]) @ components[:, 2:4]
_, C, _ = np.linalg.svd(V)
theta[0, 0] = np.degrees(np.arccos(C[0]))
theta[1, 0] = np.degrees(np.arccos(C[1]))

V = np.transpose(components[:, 2:4]) @ components[:, 4:]
_, C, _ = np.linalg.svd(V)
theta[0, 1] = np.degrees(np.arccos(C[0]))
theta[1, 1] = np.degrees(np.arccos(C[1]))

V = np.transpose(components[:, 0:2]) @ components[:, 4:]
_, C, _ = np.linalg.svd(V)
theta[0, 2] = np.degrees(np.arccos(C[0]))
theta[1, 2] = np.degrees(np.arccos(C[1]))

print("子空间夹角", theta)


n_ctrl = 1  # repeat times，可根据需要调整
rng = np.random.default_rng(seed=42)

ctrl_angles = np.zeros((n_ctrl, 3))  # 记录每次随机分割后的第一主角（deg）

# delay_mean 原先为 shape (N_SAMPLES, 1, N_HID)
# X 形状为 (N_SAMPLES, 18)
for r in range(n_ctrl):
    perm = rng.permutation(N_SAMPLES)
    idx1 = perm[: N_SAMPLES // 2]
    idx2 = perm[N_SAMPLES // 2 :]

    # fit beta on subset1 and subset2
    beta1 = np.zeros([N_HID, 18])
    beta2 = np.zeros([N_HID, 18])
    model = Lasso(alpha=0.001, max_iter=5000)
    for k in range(N_HID):
        y1 = delay_mean[idx1, 0, k]
        y2 = delay_mean[idx2, 0, k]
        model.fit(X[idx1], y1)
        beta1[k, :] = model.coef_
        model.fit(X[idx2], y2)
        beta2[k, :] = model.coef_

    # 对三个子空间分别计算第一主角（PC1-PC2 空间）
    angs = []
    for i in [1, 2, 3]:
        analy1 = np.transpose(beta1[:, (i - 1) * 6 : i * 6])  # (6, N_HID)
        analy2 = np.transpose(beta2[:, (i - 1) * 6 : i * 6])
        pca1 = PCA(n_components=2).fit(analy1)
        pca2 = PCA(n_components=2).fit(analy2)
        U1 = pca1.components_.T  # (N_HID,2)
        U2 = pca2.components_.T
        V = U1.T @ U2
        _, C, _ = np.linalg.svd(V)
        c0 = np.clip(C[0], -1.0, 1.0)
        ang = np.degrees(np.arccos(c0))
        angs.append(ang)
    ctrl_angles[r, :] = angs

ctrl_mean = ctrl_angles.mean(axis=0)
ctrl_std = ctrl_angles.std(axis=0)
# ------- 绘图：原始子空间角度与 control 同图对比 -------
xx = [1, 2, 3]
subspace_labels = ["1-2", "2-3", "1-3"]
control_labels = ["1-1", "2-2", "3-3"]

fig, ax = plt.subplots(figsize=(6, 4))

# 主数据（子空间主角）与 control 均以角度（度）为纵轴绘制
ax.plot(xx, theta[0], marker="o", color="red", lw=1.5)
ax.plot(xx, ctrl_mean, marker="s", color="gray", lw=1.5)
ax.fill_between(
    xx, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color="gray", alpha=0.25
)

# 纵轴（左）设置为灰色角度标注
ax.set_ylabel("Angle (deg)", color="gray")
ax.set_ylim(0, 90)
ax.tick_params(axis="y", colors="gray", which="both", width=1.5, direction="in")
ax.spines["left"].set_color("gray")
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_color("gray")
ax.spines["bottom"].set_linewidth(1.5)

# 底部 x 轴：显示 control 标签（灰色），并在下方添加灰色文字说明
ax.set_xticks(xx)
ax.set_xticklabels(control_labels)
ax.set_xlabel("Control", color="gray")
ax.xaxis.set_label_position("bottom")
ax.xaxis.label.set_color("gray")
ax.xaxis.set_label_coords(0.5, -0.12)
ax.tick_params(axis="x", colors="gray", direction="in")

# 右侧 y 轴：与左侧相同刻度/标签，但为红色
ax_right = ax.twinx()
ax_right.set_zorder(ax.get_zorder() + 1)
ax_right.patch.set_visible(False)
ax_right.set_ylim(ax.get_ylim())
ax_right.set_yticks(ax.get_yticks())
ax_right.set_ylabel("Angle (deg)", color="red")
ax_right.tick_params(axis="y", colors="red", which="both", width=1.5, direction="in")
ax_right.spines["right"].set_color("red")
ax_right.spines["right"].set_linewidth(1.5)

# # 顶部 x 轴：显示子空间对（红色）
ax_top = ax.twiny()
ax_top.set_zorder(ax.get_zorder() + 10)
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(xx)
ax_top.set_xticklabels(subspace_labels)
ax_top.tick_params(axis="x", colors="red", direction="in")
ax_top.spines["top"].set_color("red")
ax_top.spines["top"].set_linewidth(1.5)
# 在顶轴上方标注红色文字
ax_top.xaxis.set_label_position("top")
ax_top.xaxis.label.set_color("red")
ax_top.set_xlabel("Between subspaces", color="red", labelpad=8)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax_right.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    FIG_DIR / f"{TASK}_analysis_with_control_custom_axes.png", bbox_inches="tight"
)


def plot_delay_pca(colors1, colors2, colors3, fig_name):

    # Combine Low scatter plots (PC1-2, PC3-4, PC5-6) into one figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # compute global axis limits across all three PC pairs
    xs = np.concatenate([Low[:, 0], Low[:, 2], Low[:, 4]])
    ys = np.concatenate([Low[:, 1], Low[:, 3], Low[:, 5]])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = dx * 0.12 if dx != 0 else 0.1
    pad_y = dy * 0.12 if dy != 0 else 0.1
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    # marker/alpha settings
    marker_size = 48
    alpha_val = 0.9

    # subplot 1: PC1 vs PC2
    ax = axes[0]
    ax.scatter(
        Low[:, 0], Low[:, 1], c=colors1, marker="o", s=marker_size, alpha=alpha_val
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # subplot 2: PC3 vs PC4
    ax = axes[1]
    ax.scatter(
        Low[:, 2], Low[:, 3], c=colors2, marker="o", s=marker_size, alpha=alpha_val
    )
    ax.set_xlabel("PC3")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # subplot 3: PC5 vs PC6
    ax = axes[2]
    ax.scatter(
        Low[:, 4], Low[:, 5], c=colors3, marker="o", s=marker_size, alpha=alpha_val
    )
    ax.set_xlabel("PC5")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # add suptitle and a color legend for item labels
    fig.suptitle("Delay PCA (Low) colored by stimulus items", fontsize=16)
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
        handles=legend_handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.93)
    )

    # adjust spacing to avoid overlap and ensure legend fits
    plt.subplots_adjust(wspace=0.28, top=0.88)
    plt.savefig(
        FIG_DIR / f"{TASK}_{fig_name}.png", bbox_inches="tight", pad_inches=0.12
    )
    print(f"Saved figure: {FIG_DIR / f'{TASK}_{fig_name}.png'}")
    plt.close()


plot_delay_pca(
    colors1=colors_rank1,
    colors2=colors_rank2,
    colors3=colors_rank3,
    fig_name="pca_analysis_pre_cue_stage",
)
