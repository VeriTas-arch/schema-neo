"""
应该是正常的。
复现结束。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
hiddens = np.tanh(hiddens)


def GLM(hiddens_delay, cc):
    delay_mean = np.mean(hiddens_delay, axis=1)
    delay_mean = delay_mean[:, np.newaxis, :]  # (120, 1, 50)
    compute = delay_mean  # (120, 1, 50)

    def compute_one(class_i):
        if class_i >= 120:
            class_i = class_i - 120

        item_1, item_2, item_3 = PERMS[class_i]

        if cc == 1:
            return item_3, item_2, item_1
        else:
            return item_1, item_2, item_3

    X = np.zeros([120, 18])
    for i in np.arange(0, 120):
        x1, x2, x3 = compute_one(i)
        X[i, x1] = 1
        X[i, x2 + 6] = 1
        X[i, x3 + 12] = 1

    model = Lasso(alpha=0.001)  # alpha 是正则化参数
    beta = np.zeros([N_HID, 18])
    for i in np.arange(0, N_HID):
        model.fit(X, compute[:, :, i])
        beta[i, :] = model.coef_

    n_components_ev = 6
    components = {}
    for i in [1, 2, 3]:  # e.g., S1, S2, S3的层次
        analy = beta[:, (i - 1) * 6 : i * 6].T  # 转置为 (6, N_HID)
        # PCA for THETA (2D)
        pca_THETA = PCA(n_components=2)
        pca_THETA.fit(analy)
        components[f"pca_THETA_S{i}"] = pca_THETA.components_  # 基向量 (2, N_HID)

        # PCA for EV (6D)
        pca_EV = PCA(n_components=n_components_ev)
        pca_EV.fit(analy)
        components[f"pca_EV_S{i}"] = pca_EV.components_  # (6, N_HID)

    # 返回子空间基矩阵，不再返回投影坐标
    return components


components_D1 = GLM(hiddens[0:N_CLASS, tps["pre_cue_start"] : tps["cue_start"], :], 0)
S1D1F_subspace = components_D1["pca_THETA_S1"]
S2D1F_subspace = components_D1["pca_THETA_S3"]

components_D1 = GLM(hiddens[N_CLASS:, tps["pre_cue_start"] : tps["cue_start"], :], 0)
S1D1B_subspace = components_D1["pca_THETA_S1"]
S2D1B_subspace = components_D1["pca_THETA_S3"]

components_D2 = GLM(hiddens[0:N_CLASS, tps["cue_end"] : tps["post_cue_end"], :], 0)
S1D2F_subspace = components_D2["pca_THETA_S1"]
S2D2F_subspace = components_D2["pca_THETA_S3"]

components_D2 = GLM(hiddens[N_CLASS:, tps["cue_end"] : tps["post_cue_end"], :], 1)
S1D2B_subspace = components_D2["pca_THETA_S1"]
S2D2B_subspace = components_D2["pca_THETA_S3"]


def load_condition_data(hiddens_full, time_window, trial_indices, cc, aa):
    """
    从完整数据集中提取某个条件的神经活动和标签
    参数:
    - hiddens_full: 完整的隐状态数据 (shape: [240, 100, N_HID])
    - time_window: 时间窗口 (D1: 58:68, D2: 88:98)
    - trial_indices: 属于该条件的 trial 索引（如前向或后向）
    - cc: 规则类型 (0=前向F, 1=后向B)
    返回:
    - X_cond: 处理后的神经活动 (shape: [n_trials, N_HID])
    - y_cond: 对应的刺激位置标签 (1-6)
    """
    # 提取原始神经活动
    X_raw = hiddens_full[trial_indices, time_window, :]  # (n_trials, time_steps, N_HID)
    # 预处理（tanh + 时间平均） 已经包含tanh
    X_processed = X_raw
    X_cond = np.mean(X_processed, axis=1)  # (n_trials, N_HID)

    # 生成标签 (假设你需要解码 S1 的位置)
    def compute_one(class_i, cc):
        if class_i >= 120:
            class_i = class_i - 120

        item_1, item_2, item_3 = PERMS[class_i]

        if cc == 1:
            return item_3, item_2, item_1
        else:
            return item_1, item_2, item_3

    y_cond = []
    for i in trial_indices:  # trial索引可能需要根据条件对齐

        if aa == 0:
            s1_pos, _, _ = compute_one(i, cc)
        else:
            _, _, s1_pos = compute_one(i, cc)
        y_cond.append(s1_pos)

    return X_cond, np.array(y_cond)


X_S1D1F, y_S1D1F = load_condition_data(
    hiddens,
    time_window=slice(tps["pre_cue_start"], tps["cue_start"]),
    trial_indices=np.arange(0, 120),
    cc=0,
    aa=0,  # 前120 trials为前向
)

X_S2D1F, y_S2D1F = load_condition_data(
    hiddens,
    time_window=slice(tps["pre_cue_start"], tps["cue_start"]),
    trial_indices=np.arange(0, 120),
    cc=0,
    aa=1,  # 前120 trials为前向
)

X_S2D1B, y_S2D1B = load_condition_data(
    hiddens,
    time_window=slice(tps["pre_cue_start"], tps["cue_start"]),
    trial_indices=np.arange(120, 240),
    cc=0,
    aa=1,
)

X_S1D1B, y_S1D1B = load_condition_data(
    hiddens,
    time_window=slice(tps["pre_cue_start"], tps["cue_start"]),
    trial_indices=np.arange(120, 240),
    cc=0,
    aa=0,
)

X_S1D2F, y_S1D2F = load_condition_data(
    hiddens,
    time_window=slice(tps["cue_end"], tps["post_cue_end"]),
    trial_indices=np.arange(0, 120),
    cc=0,
    aa=0,
)

X_S2D2F, y_S2D2F = load_condition_data(
    hiddens,
    time_window=slice(tps["cue_end"], tps["post_cue_end"]),
    trial_indices=np.arange(0, 120),
    cc=0,
    aa=1,
)

X_S1D2B, y_S1D2B = load_condition_data(
    hiddens,
    time_window=slice(tps["cue_end"], tps["post_cue_end"]),
    trial_indices=np.arange(120, 240),
    cc=1,
    aa=0,
)

X_S2D2B, y_S2D2B = load_condition_data(
    hiddens,
    time_window=slice(tps["cue_end"], tps["post_cue_end"]),
    trial_indices=np.arange(120, 240),
    cc=1,
    aa=1,
)


def cross_condition_decoding(subspace_A, X_B, y_B, n_components=2, n_folds=5):
    """
    使用A子空间解码B数据的S1位置
    参数:
    - subspace_A: 子空间基矩阵 (n_components, N_HID)
    - X_B: B条件的神经数据 (n_samples, N_HID)
    - y_B: B条件的S1位置标签 (n_samples,)
    - n_components: 子空间的维度（若基矩阵未切片则不需要）
    - n_folds: 交叉验证折数
    """
    # 投影到A子空间
    X_projected = X_B @ subspace_A[:n_components, :].T  # (n_samples, n_components)

    # 标准化（根据交叉验证的折叠情况）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_projected)

    # 交叉验证评估分类性能
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    scores = cross_val_score(model, X_scaled, y_B, cv=n_folds)
    return np.mean(scores)


conditions = [
    {"name": "S1D1F", "subspace": S1D1F_subspace, "X": X_S1D1F, "y": y_S1D1F},
    {"name": "S1D1B", "subspace": S1D1B_subspace, "X": X_S1D1B, "y": y_S1D1B},
    {"name": "S1D2F", "subspace": S1D2F_subspace, "X": X_S1D2F, "y": y_S1D2F},
    {"name": "S2D2B", "subspace": S2D2B_subspace, "X": X_S2D2B, "y": y_S2D2B},
    {"name": "S2D1F", "subspace": S2D1F_subspace, "X": X_S2D1F, "y": y_S2D1F},
    {"name": "S2D1B", "subspace": S2D1B_subspace, "X": X_S2D1B, "y": y_S2D1B},
    {"name": "S2D2F", "subspace": S2D2F_subspace, "X": X_S2D2F, "y": y_S2D2F},
    {"name": "S1D2B", "subspace": S1D2B_subspace, "X": X_S1D2B, "y": y_S1D2B},
]


# 初始化准确率矩阵（n_conditions x n_conditions）
n_conds = len(conditions)
acc_matrix = np.zeros((n_conds, n_conds))

# 计算每个条件对的准确率
for i, cond_A in enumerate(conditions):
    for j, cond_B in enumerate(conditions):
        acc = cross_condition_decoding(
            subspace_A=cond_A["subspace"], X_B=cond_B["X"], y_B=cond_B["y"]
        )
        acc_matrix[i, j] = acc


# ==== 自定义颜色映射 ====
colors = [(1, 1, 1), (0.3, 0.6, 1), (0, 0, 1)]
cmap_white_blue = LinearSegmentedColormap.from_list("white_blue", colors)

# ==== 生成准确率矩阵（假设已计算完 acc_matrix） ====
labels = [cond["name"] for cond in conditions]
df_acc = pd.DataFrame(acc_matrix, index=labels, columns=labels)

# ==== 绘制热力图 ====
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    df_acc,
    annot=True,
    fmt=".2f",
    cmap=cmap_white_blue,
    vmin=0.0,
    vmax=1.0,
    linewidths=0.5,
    square=True,  # 添加方格形状
    cbar_kws={"label": "Decoding Accuracy"},
)

# 添加标签和美化
ax.set_title("Cross-Condition Decoding Accuracy", fontsize=14)
ax.set_xlabel("Test Condition (Feature Space)", fontsize=12)
ax.set_ylabel("Trained Subspace Condition", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "cross_condition_decoding_accuracy.png")
