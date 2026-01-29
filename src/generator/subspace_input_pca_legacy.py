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
labels = outputs_dict["labels"]

hiddens = np.tanh(hiddens)


def compute_one(class_i: int) -> tuple[int, int, int]:
    """
    Return the three items for a given permutation index.

    Parameters
    ----------
    class_i : int
        Index into the global ``PERMS`` permutation list.

    Returns
    -------
    tuple of int
        (item_1, item_2, item_3) corresponding to ranks 1..3.
    """
    item_1, item_2, item_3 = PERMS[class_i]
    return item_1, item_2, item_3


# 构建线性回归模型
# one-hot vector
X1, X2, X3 = [], [], []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)

colors_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]
colors_rank1 = [colors_list[i] for i in X1]
colors_rank2 = [colors_list[i] for i in X2]
colors_rank3 = [colors_list[i] for i in X3]

hiddens_input1 = hiddens[:, tps["stim1_start"] : tps["stim1_off"], :]
hiddens_input2 = hiddens[:, tps["stim2_start"] : tps["stim2_off"], :]
hiddens_input3 = hiddens[:, tps["stim3_start"] : tps["stim3_off"], :]

hiddens_input1_delay = hiddens[:, tps["stim1_off"] : tps["stim1_end"], :]
hiddens_input2_delay = hiddens[:, tps["stim2_off"] : tps["stim2_end"], :]
hiddens_input3_delay = hiddens[:, tps["stim3_off"] : tps["stim3_end"], :]

hiddens_stim1 = hiddens[:, tps["stim1_start"] : tps["stim1_end"], :]
hiddens_stim2 = hiddens[:, tps["stim2_start"] : tps["stim2_end"], :]
hiddens_stim3 = hiddens[:, tps["stim3_start"] : tps["stim3_end"], :]

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

INPUT1 = np.mean(hiddens_input1, axis=1).reshape(-1, N_HID)
INPUT2 = np.mean(hiddens_input2, axis=1).reshape(-1, N_HID)
INPUT3 = np.mean(hiddens_input3, axis=1).reshape(-1, N_HID)

INPUT1_DELAY = np.mean(hiddens_input1_delay, axis=1).reshape(-1, N_HID)
INPUT2_DELAY = np.mean(hiddens_input2_delay, axis=1).reshape(-1, N_HID)
INPUT3_DELAY = np.mean(hiddens_input3_delay, axis=1).reshape(-1, N_HID)

DELAY = np.mean(hiddens_delay, axis=1).reshape(-1, N_HID)


def set_plot():
    """
    Set plotting parameters consistent with GLM version.
    """
    plt.style.use("ggplot")

    plt.rcParams["figure.autolayout"] = True

    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["lines.markeredgewidth"] = 0.003
    plt.rcParams["lines.markersize"] = 3
    plt.rcParams["font.size"] = 14
    plt.rcParams["legend.fontsize"] = 11
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


def set_plot_legacy(ll=7):
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
            "axes.labelsize": 15,  # 坐标轴标题(X/Y轴标签)
            "xtick.labelsize": 15,  # X轴刻度标签
            "ytick.labelsize": 15,  # Y轴刻度标签
            "figure.titlesize": 15,  # 整个图标题
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


def process_single_model_pca(params):
    """
    处理单个模型的数据，计算PCA分析

    Parameters
    ----------
    params : dict
        包含模型参数和数据的字典

    Returns
    -------
    all_Lows : list
        包含三个时期PCA低维表示的列表
    all_variance : list
        包含三个时期方差解释率的列表
    """
    outputs_dict = params["OUTPUTS_DICT"]
    tps = params["TIMEPOINTS"]

    hiddens = outputs_dict["hiddens"]
    hiddens = np.tanh(hiddens)

    hiddens_stim1 = hiddens[:, tps["stim1_start"] : tps["stim1_end"], :]
    hiddens_stim2 = hiddens[:, tps["stim2_start"] : tps["stim2_end"], :]
    hiddens_stim3 = hiddens[:, tps["stim3_start"] : tps["stim3_end"], :]

    hiddens_list = [hiddens_stim1, hiddens_stim2, hiddens_stim3]

    all_Lows = []
    all_variance = []

    for hiddens_data in hiddens_list:
        Low, var_exp, _ = utils.pca_single(hiddens_data)
        all_Lows.append(Low)
        all_variance.append(var_exp)

    return all_Lows, all_variance


def plot_pca_grid_multi_models(
    params_list,
    colors,
    row_labels,
    fig_dir,
    fig_name="input_pca_multi_models",
    save_variance_subplots=False,
):
    """
    使用多个模型绘制PCA投影和方差解释率，计算均值和标准差

    Parameters
    ----------
    params_list : list
        包含多个模型参数的列表
    colors : list of list
        颜色数组用于绘制每个阶段
    row_labels : list of str
        每行的标签
    fig_dir : str
        图片保存路径
    fig_name : str, optional
        输出文件名（不含扩展名）
    save_variance_subplots : bool, optional
        是否单独保存每个方差解释率子图为PDF，默认为False
    """
    print(f"处理 {len(params_list)} 个模型...")
    n_models = len(params_list)

    # 收集所有模型的结果
    all_Lows_list = []
    all_variance_list = []

    for model_idx, params in enumerate(params_list):
        print(f"  处理模型 {model_idx + 1}/{n_models}...")
        Lows, variance = process_single_model_pca(params)
        all_Lows_list.append(Lows)
        all_variance_list.append(variance)

    # 转换为numpy数组以便计算统计量
    # all_variance_list: shape (n_models, 3, 8) - 3个时期，每个时期8个PC
    variance_array = np.array(all_variance_list)

    # 计算方差解释率的均值和标准差
    variance_mean = np.mean(variance_array, axis=0)  # shape (3, 8)
    variance_std = np.std(variance_array, axis=0, ddof=1)  # shape (3, 8)

    # 使用第一个模型的Low数据进行散点图绘制
    Lows_selected_model = all_Lows_list[1]  # shape (3, 120, 6)

    # 创建3x4大图
    fig = plt.figure(figsize=(18, 12))

    # 颜色设置
    col_labels = ["A", "B", "C"]

    # 绘制每个时期
    for row in range(3):
        col_label = col_labels[row]
        Low_row = Lows_selected_model[row]
        variance_mean_row = variance_mean[row]
        variance_std_row = variance_std[row]

        # 前3列：PCA投影（使用第一个模型的数据）
        for col in range(3):
            ax = fig.add_subplot(3, 4, row * 4 + col + 1)

            # x, y label
            ax.set_xlabel(f"PC{col * 2 + 1}", labelpad=-12)
            ax.set_ylabel(f"PC{col * 2 + 2}", labelpad=-10)

            # 绘制散点（使用第一个模型的数据）
            ax.scatter(
                Low_row[:, col * 2],
                Low_row[:, col * 2 + 1],
                c=colors[col],
                marker="o",
                alpha=0.8,
                s=80,
                zorder=2,
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

        # 第4列：方差解释率（带误差条）
        ax = fig.add_subplot(3, 4, row * 4 + 4)
        ev_colors = plt.cm.viridis(np.linspace(0.3, 0.9, 8))
        label = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
        x_pos = np.arange(len(label))

        # 绘制带有误差条的条形图
        ax.bar(
            label,
            variance_mean_row[:6],
            yerr=variance_std_row[:6],
            color=ev_colors,
            edgecolor="white",
            linewidth=1.5,
            zorder=2,
            capsize=5,
            error_kw={"linewidth": 1.5, "capthick": 1.5},
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(label)
        ax.set_yticks([0, 0.3, 0.6])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", width=1.5, direction="in", length=4)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        if row == 0:
            ax.set_title("PCA Variance Explained")

        # 单独保存方差解释率子图
        if save_variance_subplots:
            # 应用绘图风格设置
            set_plot_legacy()

            fig_var, ax_var = plt.subplots()
            ax_var.bar(
                label,
                variance_mean_row[:6],
                yerr=variance_std_row[:6],
                color=ev_colors,
                zorder=2,
                capsize=5,
                error_kw={"linewidth": 1.5, "capthick": 1.5},
            )
            ax_var.set_xticks(x_pos)
            ax_var.set_xticklabels(label)
            ax_var.set_yticks([0, 0.3, 0.6])
            ax_var.spines["top"].set_visible(False)
            ax_var.spines["right"].set_visible(False)
            ax_var.tick_params(
                axis="both", which="major", width=1, direction="in", length=4
            )
            ax_var.spines["bottom"].set_linewidth(1)
            ax_var.spines["left"].set_linewidth(1)

            plt.tight_layout()
            var_out_path = (
                Path(fig_dir) / f"{fig_name}_variance_{row_labels[row].lower()}.pdf"
            )
            plt.savefig(var_out_path, bbox_inches="tight", dpi=150)
            print(f"Variance subplot saved to {var_out_path}")
            plt.close()

        print(
            f"{row_labels[row]}: Cumulative variance (first 2 PCs): {variance_mean_row[:2].sum():.3f}"
        )

    plt.tight_layout()

    out_path = Path(fig_dir) / f"{fig_name}.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Figure saved to {out_path}")
    plt.close()

    return Lows_selected_model, variance_mean, variance_std


if __name__ == "__main__":
    layers = {
        "Input1": INPUT1,
        "Input1_D": INPUT1_DELAY,
        "Input2": INPUT2,
        "Input2_D": INPUT2_DELAY,
        "Input3": INPUT3,
        "Input3_D": INPUT3_DELAY,
    }

    # 使用多模型版本绘制PCA结果
    print("绘制三个时期的PCA分析（3x4大图，多模型平均）...")
    Low_mean, variance_mean, variance_std = plot_pca_grid_multi_models(
        params_list,
        [colors_rank1, colors_rank2, colors_rank3],
        ["Input1", "Input2", "Input3"],
        FIG_DIR,
        fig_name="sfig_input_pca",
        save_variance_subplots=True,  # 单独保存方差解释率子图
    )
    print("所有图表绘制完成！")
