import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

import generator.utils as utils
from sklearn.decomposition import PCA

#### parameters
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name
params = utils.initialize_analysis_legacy(TASK, N_SAMPLES, FILENAME, silence_index=1)

params_list = []
for i in range(5):
    params_list.append(
        utils.initialize_analysis_legacy_multi_models(
            i, TASK, N_SAMPLES, FILENAME, silence_index=1
        )
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
plt.savefig(f"{FIG_DIR}/delay_variance_length2.pdf")
print(f"saved to {FIG_DIR}/delay_variance_length2.pdf")
# plt.show()
plt.close()
