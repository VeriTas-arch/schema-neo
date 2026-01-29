"""
分析forward任务中的encoding阶段, 不同时期同角标子空间之间的关系。
例如, 希望证明stimulus 1输入时的subspace 1与stimulus 2 输入时的subspace 2仅为旋转关系。
计算不同阶段的相同角标子空间之间的旋转矩阵, 然后旋转数据, 再投影到对应PCA上。
"""

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
    params_list.append(utils.initialize_analysis_legacy_multi_models(i, TASK, N_SAMPLES, FILENAME))

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

hiddens = outputs_dict["hiddens"]
labels = outputs_dict["labels"]

hiddens = np.tanh(hiddens)

hiddens_stim1 = hiddens[:, tps["stim1_start"] : tps["stim1_end"], :]
hiddens_stim2 = hiddens[:, tps["stim2_start"] : tps["stim2_end"], :]
hiddens_stim3 = hiddens[:, tps["stim3_start"] : tps["stim3_end"], :]

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

hiddens_base = hiddens[:, 0:7, :]

mean_len = 1
delay_mean = np.mean(hiddens_delay, axis=1)
pca_x = PCA(n_components=8, svd_solver="full").fit(delay_mean)

# 构建线性回归模型
# one-hot vector
X = np.zeros([N_SAMPLES, 18])
X1 = []
X2 = []
X3 = []
for i in np.arange(0, 120):
    x1, x2, x3 = PERMS[i]
    X[i, x1] = 1
    X[i, x2 + 6] = 1
    X[i, x3 + 12] = 1
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)


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

    plt.rcParams["figure.autolayout"] = True

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

    plt.rcParams.update(
        {
            "axes.labelsize": 18,  # 坐标轴标题(X/Y轴标签)
            "xtick.labelsize": 18,  # X轴刻度标签
            "ytick.labelsize": 18,  # Y轴刻度标签
            "figure.titlesize": 18,  # 整个图标题
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


set_plot()


def plot_glm_subspace_combined(
    hiddens_stim1,
    hiddens_stim2,
    hiddens_stim3,
    fig_dir,
    fig_name=None,
    hiddens_base=None,
    enable_base=False,
):
    """
    绘制三个时期的GLM子空间分析和方差解释率（3x4大图）

    Parameters
    ----------
    hiddens_stim1 : np.ndarray
        stim1时期的hiddens数据, shape (120, time_len, N_HID)
    hiddens_stim2 : np.ndarray
        stim2时期的hiddens数据, shape (120, time_len, N_HID)
    hiddens_stim3 : np.ndarray
        stim3时期的hiddens数据, shape (120, time_len, N_HID)
    fig_dir : str
        图片保存路径
    hiddens_base : np.ndarray, optional
        baseline时期的hiddens数据, shape (120, 7, N_HID)
    """
    n_components_ev = 6
    col_labels = ["A", "B", "C"]
    hiddens_list = [hiddens_stim1, hiddens_stim2, hiddens_stim3]

    # 存储每个时期的结果
    all_h_pca = []
    all_h_pca_base = []
    all_variance = []

    if hiddens_base is not None:
        baseline_mean = np.mean(hiddens_base, axis=1)
        beta_base = np.zeros([N_HID, 18])
        for i in np.arange(0, N_HID):
            model_base = Lasso(alpha=0.001)
            model_base.fit(X, baseline_mean[:, i])
            beta_base[i, :] = model_base.coef_

    # 处理每个时期
    for idx, hiddens_data in enumerate(hiddens_list):
        # 计算平均激活
        compute = np.mean(hiddens_data, axis=1)[:, np.newaxis, :]

        # 构建线性回归模型并计算beta
        model = Lasso(alpha=0.001)
        beta = np.zeros([N_HID, 18])
        for i in np.arange(0, N_HID):
            model.fit(X, compute[:, :, i])
            beta[i, :] = model.coef_

        # PCA分析
        h_pca = np.zeros([6, n_components_ev, 3])

        for i in [1, 2, 3]:
            analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
            pca_EV = PCA(n_components=n_components_ev, svd_solver="full")
            pca_EV.fit(analy)
            h_pca[:, :, i - 1] = pca_EV.transform(analy)

        # 如果有baseline数据，计算该时期的baseline投影
        h_pca_base_period = None
        if hiddens_base is not None:
            h_pca_base_period = np.zeros([6, 2, 3])
            for i in [1, 2, 3]:
                analy_base = np.transpose(beta_base[:, (i - 1) * 6 : i * 6])
                analy_delay = np.transpose(beta[:, (i - 1) * 6 : i * 6])
                pca_THETA = PCA(n_components=2, svd_solver="full")
                pca_THETA.fit(analy_delay)
                h_pca_base_period[:, :, i - 1] = pca_THETA.transform(analy_base)

        # 自动判断是否需要翻转以确保颜色顺时针排列
        h_pca_adjusted = h_pca.copy()

        # 首先判断参考子空间（j=0）的排列顺序，确保是顺时针
        def ensure_clockwise_by_flipping(data):
            """通过翻转坐标轴确保数据点按顺时针方向排列"""
            data_copy = data.copy()

            # 计算所有点的角度（注意：arctan2(y, x)）
            angles = np.arctan2(data_copy[:, 1], data_copy[:, 0])

            # 计算连续点之间的角度变化总和
            angle_sum = 0
            for i in range(len(angles)):
                diff = angles[(i + 1) % len(angles)] - angles[i]
                # 将角度差调整到[-pi, pi]范围
                while diff <= -np.pi:
                    diff += 2 * np.pi
                while diff > np.pi:
                    diff -= 2 * np.pi
                angle_sum += diff

            # 如果角度总和为正，说明是逆时针排列，需要翻转X轴
            if angle_sum > 0:
                data_copy[:, 0] = -data_copy[:, 0]

            return data_copy

        # 对参考子空间应用顺时针调整
        h_pca_adjusted[:, :, 0] = ensure_clockwise_by_flipping(h_pca[:, :, 0])

        # 对其他子空间，通过距离比较决定翻转方式
        def need_flip(data_ref, data_current, axis=0):
            dist_original = np.sum((data_ref[:, axis] - data_current[:, axis]) ** 2)
            dist_flipped = np.sum((data_ref[:, axis] - (-data_current[:, axis])) ** 2)
            return -1 if dist_flipped < dist_original else 1

        for j in [1, 2]:
            act_x = need_flip(h_pca_adjusted[:, :, 0], h_pca[:, :, j], axis=0)
            act_y = need_flip(h_pca_adjusted[:, :, 0], h_pca[:, :, j], axis=1)
            h_pca_adjusted[:, 0, j] = h_pca[:, 0, j] * act_x
            h_pca_adjusted[:, 1, j] = h_pca[:, 1, j] * act_y

        # 计算方差解释率
        pca_variance = PCA(n_components=8, svd_solver="full").fit(
            compute[:, :, :].reshape(-1, N_HID)
        )
        explained_variance_ratio = pca_variance.explained_variance_ratio_

        all_h_pca.append(h_pca_adjusted)
        all_h_pca_base.append(h_pca_base_period)
        all_variance.append(explained_variance_ratio)

    # 创建3x4大图
    fig = plt.figure(figsize=(18, 12))

    # 颜色设置
    colors_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]
    colors_base = ["#808080"] * 6

    # 每一横排：(x, y) = (PC1, PC2)，（PC3，PC4），（PC5，PC6）

    # 绘制每个时期
    for row in range(3):
        col_label = col_labels[row]
        h_pca_adjusted = all_h_pca[row]
        h_pca_base_period = all_h_pca_base[row]
        explained_variance_ratio = all_variance[row]

        # 前3列：GLM子空间结果
        for col in range(3):
            ax = fig.add_subplot(3, 4, row * 4 + col + 1)

            # x, y label
            ax.set_xlabel(f"PC{col * 2 + 1}", labelpad=-12)
            ax.set_ylabel(f"PC{col * 2 + 2}", labelpad=-10)

            last_point = None

            # 画连接线
            for i in np.arange(n_components_ev):
                if last_point is not None:
                    ax.plot(
                        [last_point[0], h_pca_adjusted[i, 0, col]],
                        [last_point[1], h_pca_adjusted[i, 1, col]],
                        color="gray",
                        linewidth=1,
                        zorder=1,
                    )
                last_point = (h_pca_adjusted[i, 0, col], h_pca_adjusted[i, 1, col])

            # 闭合线
            if last_point is not None:
                ax.plot(
                    [last_point[0], h_pca_adjusted[0, 0, col]],
                    [last_point[1], h_pca_adjusted[0, 1, col]],
                    color="gray",
                    linewidth=1,
                    zorder=1,
                )

            # 画散点
            for i in np.arange(n_components_ev):
                ax.scatter(
                    h_pca_adjusted[i, 0, col],
                    h_pca_adjusted[i, 1, col],
                    c=colors_list[i],
                    s=250,
                    zorder=2,
                    edgecolors="white",
                    linewidths=1.5,
                )
                if h_pca_base_period is not None and enable_base:
                    ax.scatter(
                        h_pca_base_period[i, 0, col],
                        h_pca_base_period[i, 1, col],
                        c=colors_base[i],
                        s=100,
                        zorder=2,
                        edgecolors="none",
                        alpha=0.3,
                    )

            ax.set_xlim(-3, 3)
            ax.set_xticks([-3, 0, 3], labels=["-3", "", "3"])
            ax.set_ylim(-3, 3)
            ax.set_yticks([-3, 0, 3], labels=["", "", "3"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(
                axis="both", which="major", width=1, direction="in", length=4
            )
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["left"].set_linewidth(1)

            # 添加标题和标签
            if row == 0:
                ax.set_title(f"Rank-{col + 1} subspace")
            if col == 0:
                ax.text(
                    -0.1,
                    1.05,
                    f"{col_label}",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                    fontsize=18,
                )

        # 第4列：方差解释率
        ax = fig.add_subplot(3, 4, row * 4 + 4)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, 8))
        label = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
        ax.bar(
            label,
            explained_variance_ratio[:6],
            color=colors,
            edgecolor="white",
            linewidth=1.5,
            zorder=2,
        )
        ax.set_yticks([0, 0.3, 0.6])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", width=1.5, direction="in", length=4)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        if row == 0:
            ax.set_title("PCA Variance Explained")

    plt.tight_layout()

    fig_name = "glm_subspace_combined" if fig_name is None else fig_name

    plt.savefig(f"{fig_dir}/{fig_name}.pdf")
    print(f"Figure saved to {fig_dir}/{fig_name}.pdf")
    plt.close()

    return all_h_pca, all_variance


def process_single_model(params):
    """
    处理单个模型的数据，计算GLM子空间分析

    Parameters
    ----------
    params : dict
        包含模型参数和数据的字典

    Returns
    -------
    all_h_pca : list
        包含三个时期GLM子空间PCA结果的列表
    all_variance : list
        包含三个时期方差解释率的列表
    """
    N_HID = params["N_HID"]
    PERMS = params["PERMS"]
    outputs_dict = params["OUTPUTS_DICT"]
    tps = params["TIMEPOINTS"]

    hiddens = outputs_dict["hiddens"]

    hiddens = np.tanh(hiddens)

    hiddens_stim1 = hiddens[:, tps["stim1_start"] : tps["stim1_end"], :]
    hiddens_stim2 = hiddens[:, tps["stim2_start"] : tps["stim2_end"], :]
    hiddens_stim3 = hiddens[:, tps["stim3_start"] : tps["stim3_end"], :]
    hiddens_base = hiddens[:, 0:7, :]

    # 构建线性回归模型
    N_SAMPLES = 120
    X = np.zeros([N_SAMPLES, 18])
    for i in np.arange(0, 120):
        x1, x2, x3 = PERMS[i]
        X[i, x1] = 1
        X[i, x2 + 6] = 1
        X[i, x3 + 12] = 1

    # baseline数据处理
    baseline_mean = np.mean(hiddens_base, axis=1)
    beta_base = np.zeros([N_HID, 18])
    model_base = Lasso(alpha=0.001)
    for i in np.arange(0, N_HID):
        model_base.fit(X, baseline_mean[:, i])
        beta_base[i, :] = model_base.coef_

    hiddens_list = [hiddens_stim1, hiddens_stim2, hiddens_stim3]
    n_components_ev = 6

    all_h_pca = []
    all_variance = []

    for hiddens_data in hiddens_list:
        # 计算平均激活
        compute = np.mean(hiddens_data, axis=1)[:, np.newaxis, :]

        # 构建线性回归模型并计算beta
        model = Lasso(alpha=0.001)
        beta = np.zeros([N_HID, 18])
        for i in np.arange(0, N_HID):
            model.fit(X, compute[:, :, i])
            beta[i, :] = model.coef_

        # PCA分析
        h_pca = np.zeros([6, n_components_ev, 3])

        for i in [1, 2, 3]:
            analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
            pca_EV = PCA(n_components=n_components_ev, svd_solver="full")
            pca_EV.fit(analy)
            h_pca[:, :, i - 1] = pca_EV.transform(analy)

        # 自动判断是否需要翻转以确保颜色顺时针排列
        h_pca_adjusted = h_pca.copy()

        def ensure_clockwise_by_flipping(data):
            """通过翻转坐标轴确保数据点按顺时针方向排列"""
            data_copy = data.copy()
            angles = np.arctan2(data_copy[:, 1], data_copy[:, 0])
            angle_sum = 0
            for i in range(len(angles)):
                diff = angles[(i + 1) % len(angles)] - angles[i]
                while diff <= -np.pi:
                    diff += 2 * np.pi
                while diff > np.pi:
                    diff -= 2 * np.pi
                angle_sum += diff
            if angle_sum > 0:
                data_copy[:, 0] = -data_copy[:, 0]
            return data_copy

        h_pca_adjusted[:, :, 0] = ensure_clockwise_by_flipping(h_pca[:, :, 0])

        def need_flip(data_ref, data_current, axis=0):
            dist_original = np.sum((data_ref[:, axis] - data_current[:, axis]) ** 2)
            dist_flipped = np.sum((data_ref[:, axis] - (-data_current[:, axis])) ** 2)
            return -1 if dist_flipped < dist_original else 1

        for j in [1, 2]:
            act_x = need_flip(h_pca_adjusted[:, :, 0], h_pca[:, :, j], axis=0)
            act_y = need_flip(h_pca_adjusted[:, :, 0], h_pca[:, :, j], axis=1)
            h_pca_adjusted[:, 0, j] = h_pca[:, 0, j] * act_x
            h_pca_adjusted[:, 1, j] = h_pca[:, 1, j] * act_y

        # 计算方差解释率
        pca_variance = PCA(n_components=8, svd_solver="full").fit(
            compute[:, :, :].reshape(-1, N_HID)
        )
        explained_variance_ratio = pca_variance.explained_variance_ratio_

        all_h_pca.append(h_pca_adjusted)
        all_variance.append(explained_variance_ratio)

    return all_h_pca, all_variance


def plot_glm_subspace_combined_multi_models(
    params_list, fig_dir, fig_name=None, enable_base=False
):
    """
    使用多个模型绘制GLM子空间分析和方差解释率，计算均值和标准差

    Parameters
    ----------
    params_list : list
        包含多个模型参数的列表
    fig_dir : str
        图片保存路径
    fig_name : str, optional
        图片文件名
    enable_base : bool, optional
        是否显示baseline数据
    """
    print(f"处理 {len(params_list)} 个模型...")
    n_models = len(params_list)

    # 收集所有模型的结果
    all_h_pca_list = []
    all_variance_list = []

    for model_idx, params in enumerate(params_list):
        print(f"  处理模型 {model_idx + 1}/{n_models}...")
        h_pca, variance = process_single_model(params)
        all_h_pca_list.append(h_pca)
        all_variance_list.append(variance)

    # 转换为numpy数组以便计算统计量
    # all_variance_list: shape (n_models, 3, 8) - 3个时期，每个时期8个PC
    variance_array = np.array(all_variance_list)

    # 计算均值和标准差
    variance_mean = np.mean(variance_array, axis=0)  # shape (3, 8)
    variance_std = np.std(variance_array, axis=0, ddof=1)  # shape (3, 8)

    # 使用第一个模型的GLM子空间结果作为可视化（取平均值可能不太有意义）
    # 或者可以计算所有模型GLM子空间坐标的平均值
    h_pca_array = np.array(all_h_pca_list)  # shape (n_models, 3, 6, 6, 3)
    h_pca_mean = h_pca_array[1]  # model index: 1107

    # 创建3x4大图
    fig = plt.figure(figsize=(12, 12))

    # 颜色设置
    colors_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]
    col_labels = ["A", "B", "C"]
    n_components_ev = 6

    # 绘制每个时期
    for row in range(3):
        col_label = col_labels[row]
        h_pca_adjusted = h_pca_mean[row]
        # explained_variance_mean = variance_mean[row]
        # explained_variance_std = variance_std[row]

        # 前3列：GLM子空间结果
        for col in range(3):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1)

            # x, y label in consistent with (Xie, 2022)
            ax.set_xlabel("rPC1", labelpad=-12)
            ax.set_ylabel("rPC2", labelpad=-10)

            last_point = None

            # 画连接线
            for i in np.arange(n_components_ev):
                if last_point is not None:
                    ax.plot(
                        [last_point[0], h_pca_adjusted[i, 0, col]],
                        [last_point[1], h_pca_adjusted[i, 1, col]],
                        color="gray",
                        linewidth=1,
                        zorder=1,
                    )
                last_point = (h_pca_adjusted[i, 0, col], h_pca_adjusted[i, 1, col])

            # 闭合线
            if last_point is not None:
                ax.plot(
                    [last_point[0], h_pca_adjusted[0, 0, col]],
                    [last_point[1], h_pca_adjusted[0, 1, col]],
                    color="gray",
                    linewidth=1,
                    zorder=1,
                )

            # 画散点
            for i in np.arange(n_components_ev):
                ax.scatter(
                    h_pca_adjusted[i, 0, col],
                    h_pca_adjusted[i, 1, col],
                    c=colors_list[i],
                    s=250,
                    zorder=2,
                    edgecolors="white",
                    linewidths=1.5,
                )

            ax.set_xlim(-3, 3)
            ax.set_xticks([-3, 0, 3], labels=["-3", "", "3"])
            ax.set_ylim(-3, 3)
            ax.set_yticks([-3, 0, 3], labels=["", "", "3"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(
                axis="both", which="major", width=1, direction="in", length=4
            )
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["left"].set_linewidth(1)

            # 添加标题和标签
            if row == 0:
                ax.set_title(f"Rank-{col + 1} subspace")
            if col == 0:
                ax.text(
                    -0.15,
                    1.05,
                    f"{col_label}",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                    fontsize=18,
                )

        #! 注意，这里只是为了好看而绘制的方差解释率。但事实上这是对 PCA 而言的，对于 GLM 没有用，因此不应该出现在这篇图里。
        # 第4列：方差解释率（带误差条）
        # ax = fig.add_subplot(3, 4, row * 4 + 4)
        # colors = plt.cm.viridis(np.linspace(0.3, 0.9, 8))
        # label = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]

        # # 绘制带有误差条的条形图
        # ax.bar(
        #     label,
        #     explained_variance_mean[:6],
        #     yerr=explained_variance_std[:6],
        #     color=colors,
        #     edgecolor="white",
        #     linewidth=1.5,
        #     zorder=2,
        #     capsize=5,
        #     error_kw={"linewidth": 1.5, "capthick": 1.5},
        # )

        # ax.set_yticks([0, 0.3, 0.6])
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.tick_params(axis="both", which="major", width=1.5, direction="in", length=4)
        # ax.spines["bottom"].set_linewidth(1)
        # ax.spines["left"].set_linewidth(1)
        # if row == 0:
        #     ax.set_title("PCA Variance Explained")

    plt.tight_layout()

    fig_name = "sfig_input_glm_multi_models" if fig_name is None else fig_name

    plt.savefig(f"{fig_dir}/{fig_name}.pdf")
    print(f"Figure saved to {fig_dir}/{fig_name}.pdf")
    plt.close()

    return h_pca_mean, variance_mean, variance_std


# 使用新函数绘制三个时期的GLM子空间分析（3x4大图，多模型平均）
print("绘制三个时期的GLM子空间分析（3x4大图，多模型平均）...")
h_pca_mean, variance_mean, variance_std = plot_glm_subspace_combined_multi_models(
    params_list,
    FIG_DIR,
    fig_name="sfig_input_glm",
)
print("所有图表绘制完成！")
