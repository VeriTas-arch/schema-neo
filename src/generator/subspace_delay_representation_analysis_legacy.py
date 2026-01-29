import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

import generator.utils as utils
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

#### parameters
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name
params = utils.initialize_analysis_legacy(TASK, N_SAMPLES, FILENAME)

params_list = []
for i in range(5):
    params_list.append(
        utils.initialize_analysis_legacy_multi_models(i, TASK, N_SAMPLES, FILENAME)
    )

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

hiddens = outputs_dict["hiddens"]

hiddens = np.tanh(hiddens)

hiddens_stim1 = hiddens[:, tps["stim1_start"] : tps["stim1_end"], :]
hiddens_stim2 = hiddens[:, tps["stim2_start"] : tps["stim2_end"], :]
hiddens_stim3 = hiddens[:, tps["stim3_start"] : tps["stim3_end"], :]

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]


def set_plot(ll=7):
    """
    Set plotting parameters. Returns colors for plots

    Parameters
    ----------
    ll : int, optional
        Number of colors. 5 or 7. The default is 7.

    Returns
    -------
    clS : colors

    """
    plt.style.use("ggplot")

    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2  # height in inches
    fig_size = [fig_width, fig_height]
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.autolayout"] = True

    # 禁用网格
    plt.rcParams["axes.grid"] = False

    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["lines.markeredgewidth"] = 0.003
    plt.rcParams["lines.markersize"] = 3
    plt.rcParams["font.size"] = 14  # 9
    plt.rcParams["legend.fontsize"] = 11  # 7.
    plt.rcParams["axes.facecolor"] = "1"
    plt.rcParams["axes.edgecolor"] = "0"
    plt.rcParams["axes.linewidth"] = "0.7"

    plt.rcParams["axes.labelcolor"] = "0"
    plt.rcParams["axes.labelsize"] = 14  # 9
    plt.rcParams["xtick.labelsize"] = 11  # 7
    plt.rcParams["ytick.labelsize"] = 11  # 7
    plt.rcParams["xtick.color"] = "0"
    plt.rcParams["ytick.color"] = "0"
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2

    plt.rcParams["font.sans-serif"] = "Arial"

    plt.rcParams.update(
        {
            "axes.labelsize": 14,  # 坐标轴标题(X/Y轴标签)
            "xtick.labelsize": 14,  # X轴刻度标签
            "ytick.labelsize": 14,  # Y轴刻度标签
            "figure.titlesize": 14,  # 整个图标题
        }
    )

    clS = np.zeros((ll, 3))

    cl11 = np.array((102, 153, 255)) / 255.0
    cl12 = np.array((53, 153, 53)) / 255.0

    cl21 = np.array((255, 204, 51)) / 255.0
    cl22 = np.array((204, 0, 0)) / 255.0

    if ll == 7:
        clS[0, :] = 0.4 * np.ones((3,))

        clS[1, :] = cl11
        clS[2, :] = 0.5 * cl11 + 0.5 * cl12
        clS[3, :] = cl12

        clS[4, :] = cl21
        clS[5, :] = 0.5 * cl21 + 0.5 * cl22
        clS[6, :] = cl22

        clS = clS[1:]
        clS = clS[::-1]

        c2 = [67 / 256, 90 / 256, 162 / 256]
        c1 = [220 / 256, 70 / 256, 51 / 256]
        clS[0, :] = c1
        clS[5, :] = c2
    elif ll == 5:
        clS[0, :] = 0.4 * np.ones((3,))

        clS[2, :] = cl12

        clS[3, :] = cl21

        clS[4, :] = cl22
    return clS


delay_mean = np.mean(hiddens_delay, axis=1)
pca_x = PCA(n_components=8).fit(delay_mean)


def pca_different(hiddens1):
    length = hiddens1.shape[1]
    delay_mean = np.mean(hiddens1, axis=1).reshape(-1, N_HID)

    pca_x = PCA(n_components=8).fit(delay_mean)
    Low = np.mean(
        pca_x.transform(hiddens1.reshape(-1, N_HID)).reshape(120, length, 8), axis=1
    )
    # Low=pca_x.transform(delay_mean)
    var_explained = pca_x.explained_variance_ratio_
    return Low, var_explained


def process_delay_angles_single_model(params, control_split_idx=None):
    """
    处理单个模型的delay数据，计算子空间夹角

    Parameters
    ----------
    params : dict
        包含模型参数和数据的字典
    control_split_idx : tuple of two arrays, optional
        用于control计算的随机分割索引 (idx1, idx2)

    Returns
    -------
    theta : np.ndarray
        不同子空间之间的角度，shape (2, 3)
        theta[0] 是第一主向量之间的角度，theta[1] 是第二主向量之间的角度
    theta_control : np.ndarray
        control角度（同一时期同一rank之间），shape (2, 3)
    """
    outputs_dict = params["OUTPUTS_DICT"]
    tps = params["TIMEPOINTS"]
    PERMS = params["PERMS"]
    N_HID = params["N_HID"]
    N_SAMPLES = 120

    hiddens = outputs_dict["hiddens"]
    hiddens = np.tanh(hiddens)

    hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]
    delay_mean = np.mean(hiddens_delay, axis=1)[:, np.newaxis, :]

    # 构建one-hot向量
    X = np.zeros([N_SAMPLES, 18])
    for i in np.arange(0, N_SAMPLES):
        x1, x2, x3 = PERMS[i]
        X[i, x1] = 1
        X[i, x2 + 6] = 1
        X[i, x3 + 12] = 1

    # 线性回归
    model = Lasso(alpha=0.001)
    beta = np.zeros([N_HID, 18])
    for i in np.arange(0, N_HID):
        model.fit(X, delay_mean[:, :, i])
        beta[i, :] = model.coef_

    # PCA分析
    components = np.zeros([N_HID, 6])
    for i in [1, 2, 3]:
        analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
        pca_THETA = PCA(n_components=2)
        pca_THETA.fit(analy)
        components[:, (i - 1) * 2 : i * 2] = pca_THETA.components_.T

    # 计算不同子空间夹角
    theta = np.zeros([2, 3])

    # 1-2
    V = np.transpose(components[:, 0:2]) @ components[:, 2:4]
    _, C, _ = np.linalg.svd(V)
    theta[0, 0] = np.degrees(np.arccos(C[0]))
    theta[1, 0] = np.degrees(np.arccos(C[1]))

    # 2-3
    V = np.transpose(components[:, 2:4]) @ components[:, 4:]
    _, C, _ = np.linalg.svd(V)
    theta[0, 1] = np.degrees(np.arccos(C[0]))
    theta[1, 1] = np.degrees(np.arccos(C[1]))

    # 1-3
    V = np.transpose(components[:, 0:2]) @ components[:, 4:]
    _, C, _ = np.linalg.svd(V)
    theta[0, 2] = np.degrees(np.arccos(C[0]))
    theta[1, 2] = np.degrees(np.arccos(C[1]))

    # 计算control角度（使用随机分割数据的方式）
    # 如果没有提供分割索引，则使用固定种子生成
    if control_split_idx is None:
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(N_SAMPLES)
        idx1 = perm[: N_SAMPLES // 2]
        idx2 = perm[N_SAMPLES // 2 :]
    else:
        idx1, idx2 = control_split_idx

    ctrl_angles = np.zeros((1, 3))  # 记录第一主角（deg）

    # 在两个子集上分别拟合 beta
    beta1 = np.zeros([N_HID, 18])
    beta2 = np.zeros([N_HID, 18])
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
        U1 = pca1.components_.T  # (N_HID, 2)
        U2 = pca2.components_.T
        V = U1.T @ U2
        _, C, _ = np.linalg.svd(V)
        c0 = np.clip(C[0], -1.0, 1.0)
        ang = np.degrees(np.arccos(c0))
        angs.append(ang)
    ctrl_angles[0, :] = angs

    # 为了保持返回形状一致，我们用 theta_control[0] 存储第一主角，theta_control[1] 存 0
    theta_control = np.zeros([2, 3])
    theta_control[0, :] = ctrl_angles[0, :]

    return theta, theta_control


def process_delay_variance_single_model(params):
    """
    处理单个模型的delay数据，计算方差解释率

    Parameters
    ----------
    params : dict
        包含模型参数和数据的字典

    Returns
    -------
    explained_variance_ratio : np.ndarray
        方差解释率数组，shape (8,)
    """
    outputs_dict = params["OUTPUTS_DICT"]
    tps = params["TIMEPOINTS"]

    hiddens = outputs_dict["hiddens"]
    hiddens = np.tanh(hiddens)

    hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

    # 计算方差解释率
    _, explained_variance_ratio = pca_different(hiddens_delay)

    return explained_variance_ratio


def plot_delay_angles_multi_models(params_list, fig_dir, fig_name="delay_angles"):
    """
    使用多个模型绘制delay时期的子空间夹角，计算均值和标准差

    Parameters
    ----------
    params_list : list
        包含多个模型参数的列表
    fig_dir : str
        图片保存路径
    fig_name : str, optional
        输出文件名（不含扩展名）
    """
    print(f"处理 {len(params_list)} 个模型的子空间夹角...")
    n_models = len(params_list)
    N_SAMPLES = 120

    # 为每个模型生成不同的随机分割索引（使用不同种子）
    control_splits = []
    for model_idx in range(n_models):
        rng = np.random.default_rng(seed=100 + model_idx)  # 使用不同种子
        perm = rng.permutation(N_SAMPLES)
        idx1 = perm[: N_SAMPLES // 2]
        idx2 = perm[N_SAMPLES // 2 :]
        control_splits.append((idx1, idx2))

    # 收集所有模型的夹角数据
    all_theta_list = []
    all_theta_control_list = []

    for model_idx, params in enumerate(params_list):
        print(f"  处理模型 {model_idx + 1}/{n_models}...")
        theta, theta_control = process_delay_angles_single_model(
            params, control_split_idx=control_splits[model_idx]
        )
        all_theta_list.append(theta)
        all_theta_control_list.append(theta_control)

    # 转换为numpy数组
    # all_theta_list: shape (n_models, 2, 3)
    theta_array = np.array(all_theta_list)
    theta_control_array = np.array(all_theta_control_list)

    # 计算均值和标准差
    theta_mean = np.mean(theta_array, axis=0)  # shape (2, 3)
    theta_std = np.std(theta_array, axis=0, ddof=1)  # shape (2, 3)

    theta_control_mean = np.mean(theta_control_array, axis=0)  # shape (2, 3)
    theta_control_std = np.std(theta_control_array, axis=0, ddof=1)  # shape (2, 3)

    print(f"Between angles (first PC): {theta_mean[0]}")
    print(f"Between angles (first PC) std: {theta_std[0]}")
    print(f"Between angles (second PC): {theta_mean[1]}")
    print(f"Control angles (first PC): {theta_control_mean[0]}")
    print(f"Control angles (first PC) std: {theta_control_std[0]}")
    print(f"Control angles (second PC): {theta_control_mean[1]}")

    return theta_mean, theta_std, theta_control_mean, theta_control_std


def plot_delay_variance_multi_models(params_list, fig_dir, fig_name="delay_variance"):
    """
    使用多个模型绘制delay时期的方差解释率，计算均值和标准差

    Parameters
    ----------
    params_list : list
        包含多个模型参数的列表
    fig_dir : str
        图片保存路径
    fig_name : str, optional
        输出文件名（不含扩展名）
    """
    print(f"处理 {len(params_list)} 个模型的方差解释率...")
    n_models = len(params_list)

    # 收集所有模型的方差解释率
    all_variance_list = []

    for model_idx, params in enumerate(params_list):
        print(f"  处理模型 {model_idx + 1}/{n_models}...")
        explained_variance_ratio = process_delay_variance_single_model(params)
        all_variance_list.append(explained_variance_ratio)

    # 转换为numpy数组
    # all_variance_list: shape (n_models, 8) - 每个模型8个PC
    variance_array = np.array(all_variance_list)

    # 计算均值和标准差
    variance_mean = np.mean(variance_array, axis=0)  # shape (8,)
    variance_std = np.std(variance_array, axis=0, ddof=1)  # shape (8,)

    print(f"方差解释率均值: {variance_mean[:6]}")
    print(f"方差解释率标准差: {variance_std[:6]}")

    return variance_mean, variance_std


set_plot(1)
### ========== 绘制子空间夹角折线图 ==========
# 使用多模型版本计算夹角
theta_mean, theta_std, theta_control_mean, theta_control_std = (
    plot_delay_angles_multi_models(params_list, FIG_DIR, fig_name="delay_angles")
)

xx = [1, 2, 3]
subspace_labels = ["1-2", "2-3", "1-3"]
control_labels = ["1-1", "2-2", "3-3"]

fig, ax = plt.subplots()

# 主数据（子空间主角）与 control 均以角度（度）为纵轴绘制，带误差条
ax.errorbar(
    xx,
    theta_mean[0],
    yerr=theta_std[0],
    marker="s",
    color="#E74C3C",
    lw=1.5,
    markersize=4,
    elinewidth=2,
    capsize=4,
    capthick=1.5,
)
ax.errorbar(
    xx,
    theta_control_mean[0],
    yerr=theta_control_std[0],
    marker="s",
    color="gray",
    lw=1.5,
    markersize=4,
    elinewidth=2,
    capsize=4,
    capthick=1.5,
    ecolor="gray",  # 确保误差条颜色与线条一致
)

# 纵轴（左）设置为灰色角度标注
ax.set_xlim(0.7, 3.3)
ax.set_ylim(-5, 90)
ax.set_yticks([0, 30, 60, 90])
ax.set_yticklabels(["0", "30", "60", "90"])
ax.tick_params(axis="y", colors="gray", which="both", width=1, direction="in")
ax.spines["left"].set_color("gray")
ax.spines["left"].set_linewidth(1)
ax.spines["bottom"].set_color("gray")
ax.spines["bottom"].set_linewidth(1)

# 底部 x 轴：显示 control 标签（灰色）
ax.set_xticks(xx)
ax.set_xticklabels(control_labels)

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
ax_right.set_yticklabels(["0", "30", "60", "90"])

ax_right.tick_params(axis="y", colors="#E74C3C", which="both", width=1, direction="in")
ax_right.spines["left"].set_color("gray")
ax_right.spines["bottom"].set_color("gray")
ax_right.spines["right"].set_color("#E74C3C")
ax_right.spines["right"].set_linewidth(1)

# 顶部 x 轴：显示子空间对（红色）
ax_top = ax.twiny()
ax_top.set_zorder(ax.get_zorder() + 10)
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(xx)
ax_top.set_xticklabels(subspace_labels)
ax_top.tick_params(axis="x", colors="#E74C3C", direction="in")
ax_top.spines["left"].set_color("gray")
ax_top.spines["bottom"].set_color("gray")
ax_top.spines["top"].set_color("#E74C3C")
ax_top.spines["top"].set_linewidth(1)
ax_top.xaxis.set_label_position("top")
ax_top.xaxis.label.set_color("#E74C3C")

# 隐藏多余边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax_right.spines["top"].set_visible(False)
ax_right.spines["left"].set_visible(False)  # 隐藏ax_right的left spine
ax_top.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_visible(False)  # 隐藏ax_top的bottom spine

# 重新设置原始轴的left和bottom spines颜色（被共享轴覆盖后需要重新设置）
ax.spines["left"].set_color("gray")
ax.spines["bottom"].set_color("gray")

# 添加水平参考线
ax.axhline(y=45, color="gray", linestyle="--", linewidth=1, alpha=0.5)

# 确保网格被禁用
ax.grid(False)
ax_right.grid(False)
ax_top.grid(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/delay_angles.pdf")
print(f"saved to {FIG_DIR}/delay_angles.pdf")
# plt.show()
plt.close()


## ========= 绘制方差解释率分布图 ==========
plt.figure()
set_plot(1)

label = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
# 使用多模型版本计算方差解释率
variance_mean, variance_std = plot_delay_variance_multi_models(
    params_list, FIG_DIR, fig_name="delay_variance"
)

# 创建渐变色彩方案
colors = plt.cm.viridis(np.linspace(0.3, 0.9, 8))  # 使用viridis配色方案
x_pos = np.arange(len(label))

# 绘制带有误差条的柱状图
bars = plt.bar(
    x_pos,
    variance_mean[:6],
    yerr=variance_std[:6],
    color=colors,
    capsize=5,
    error_kw={"linewidth": 1.5, "capthick": 1.5},
)

plt.xticks(x_pos, label)
plt.yticks([0, 0.3, 0.6])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", which="major", width=1, direction="in", length=4)

ax = plt.gca()
ax.spines["bottom"].set_linewidth(1)
ax.spines["left"].set_linewidth(1)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/delay_variance.pdf")
print(f"saved to {FIG_DIR}/delay_variance.pdf")
# plt.show()
plt.close()
