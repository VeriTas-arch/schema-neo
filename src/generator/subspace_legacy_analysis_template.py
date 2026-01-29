import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

import generator.utils as utils

#### parameters
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name
params = utils.initialize_analysis_legacy(TASK, N_SAMPLES, FILENAME)

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]

batch = outputs_dict["batch"]
labels = outputs_dict["labels"]
hiddens = outputs_dict["hiddens"]

hiddens = np.tanh(hiddens)

tps = params["TIMEPOINTS"]

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]


#! --------- NOW YOU CAN USE THE SAME CODES AS CURRENT ONES ------------

mean_len = 1
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


delay_mean = delay_mean[:, np.newaxis, :]  # (120, 1, 50)

compute = delay_mean  # (120, 1, 50)
compute_len = mean_len


labels = labels.reshape(-1)


def compute_one(class_i):
    if class_i >= 120:
        class_i = class_i - 120
        cc = 1
    else:
        cc = 0
    item_1 = class_i // 20
    cue23 = class_i % 20
    cue2 = cue23 // 4
    cue3 = cue23 % 4
    p = [0, 1, 2, 3, 4, 5]
    p.remove(item_1)
    item_2 = p[cue2]
    p.remove(item_2)
    item_3 = p[cue3]
    if cc == 1:
        return item_1, item_2, item_3
    else:
        return item_1, item_2, item_3


# 构建线性回归模型
# one-hot vector
X = np.zeros([N_SAMPLES, 18])
X1 = []
X2 = []
X3 = []
for i in np.arange(0, 120):
    x1, x2, x3 = compute_one(i)
    X[i, x1] = 1
    X[i, x2 + 6] = 1
    X[i, x3 + 12] = 1
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)
# coefficient:beta

model = Lasso(alpha=0.001)  # alpha 是正则化参数
# model = LinearRegression()
# model = PLSRegression()
beta = np.zeros([N_HID, 18])
for i in np.arange(0, N_HID):
    model.fit(X, compute[:, :, i])
    beta[i, :] = model.coef_
# print(beta.shape)
# raise ValueError(beta)

# print(beta.shape)
## pca analysis
beta_1 = np.transpose(beta[:, 0:6])
beta_2 = np.transpose(beta[:, 6:12])
beta_3 = np.transpose(beta[:, 12:])
n_components_ev = 6
components = np.zeros([N_HID, 6])
accu_ev = np.zeros([n_components_ev, 3])
h_pca = np.zeros([6, n_components_ev, 3])
# V = np.zeros([2, 6])


for i in [1, 2, 3]:
    analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
    pca_THETA = PCA(n_components=2)
    pca_EV = PCA(n_components=n_components_ev)
    pca_THETA.fit(analy)
    components[:, (i - 1) * 2 : i * 2] = pca_THETA.components_.T

    pca_EV.fit(analy)
    explained_variance_ratio = pca_EV.explained_variance_ratio_
    num_components = len(explained_variance_ratio)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    accu_ev[:, i - 1] = cumulative_variance
    h_pca[:, :, i - 1] = pca_EV.transform(analy)


### 计算不同子空间夹角
theta = np.zeros([2, 3])
V = np.transpose(components[:, 0:2]) @ components[:, 2:4]
# raise ValueError(V.shape)
_, C, _ = np.linalg.svd(V)
# raise ValueError(C)
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


print("子空间夹角（1-2，2-3，1-3）：")
xx = [1, 2, 3]
print(theta)
plt.plot(xx, theta[0])
plt.xticks(xx)
plt.yticks()
# 设置纵坐标范围
plt.ylim(0, 90)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
plt.show()

alpha = 0.1
colors = plt.cm.viridis(np.linspace(0.1, 0.9, N_CLASS))
label = [1, 2, 3, 4, 5, 6]
for j in [0, 1, 2]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    last_point = None
    for i in np.arange(n_components_ev):
        # l_idx = labels[i]
        # ax.scatter(h_pca[i, 0], h_pca[i, 1], c=colors[l_idx], alpha=0.5)
        ax.scatter(h_pca[i, 0, j], h_pca[i, 1, j], label=label[i], s=70)
        if last_point is not None:
            ax.plot(
                [last_point[0], h_pca[i, 0, j]], [last_point[1], h_pca[i, 1, j]], "gray"
            )  # 'k-' 表示黑色实线
        last_point = (h_pca[i, 0, j], h_pca[i, 1, j])
    if last_point is not None:
        ax.plot(
            [last_point[0], h_pca[0, 0, j]], [last_point[1], h_pca[0, 1, j]], "gray"
        )

    ax.set_xlabel("PC 1")
    # ax.set_xlim(-1.5, 2.5)
    ax.set_ylabel("PC 2")
    # ax.set_ylim(-2, 2)
    ax.legend(loc="best")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
##     plt.savefig(save_dir + "/pca2_rank" + str(j+1) + ".png")


# trajectory
act_x = hiddens[:, 0:120, :].reshape(
    -1, N_HID
)  # (batch, T, N_HID) = (120, 60, 50) —— (batch*T, N_HID)
# raise ValueError(act_x.shape)
n_components = 30
pca_y = PCA(n_components=n_components).fit(act_x)
h_hid_pca = pca_y.transform(act_x)


h_hid_pca = h_hid_pca.reshape(batch, int(h_hid_pca.shape[0] / batch), n_components)


# h_hid_pca =h_hid_pca-np.mean(h_hid_pca,axis=0)

# raise ValueError(h_hid_pca.shape)  # (batch, T, n_components)
fig = plt.figure()
ax_3D = fig.add_subplot(111, projection="3d")

for i in np.arange(1):
    if i == 0:
        ax_3D.plot(
            h_hid_pca[i, 0:115, 0],
            h_hid_pca[i, 0:115, 1],
            h_hid_pca[i, 0:115, 2],
            linewidth=2,
        )
        plt.plot(
            h_hid_pca[i, 16, 0],
            h_hid_pca[i, 16, 1],
            h_hid_pca[i, 16, 2],
            marker="*",
            markersize=10,
        )
        plt.plot(
            h_hid_pca[i, 32, 0],
            h_hid_pca[i, 32, 1],
            h_hid_pca[i, 32, 2],
            marker="*",
            markersize=10,
        )
        plt.plot(
            h_hid_pca[i, 48, 0],
            h_hid_pca[i, 48, 1],
            h_hid_pca[i, 48, 2],
            marker="*",
            markersize=10,
        )

        plt.plot(
            h_hid_pca[i, 90, 0],
            h_hid_pca[i, 90, 1],
            h_hid_pca[i, 90, 2],
            marker="o",
            markersize=10,
        )

        plt.plot(
            h_hid_pca[i, 102, 0],
            h_hid_pca[i, 102, 1],
            h_hid_pca[i, 102, 2],
            marker="o",
            markersize=10,
        )

        plt.plot(
            h_hid_pca[i, 114, 0],
            h_hid_pca[i, 114, 1],
            h_hid_pca[i, 114, 2],
            marker="o",
            markersize=10,
        )

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)

colors_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]
# 深红 → 橙 → 深黄 → 深绿 → 蓝 → 亮紫
colors1 = [colors_list[i] for i in X1]
colors2 = [colors_list[i] for i in X2]
colors3 = [colors_list[i] for i in X3]


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


set_plot(1)

Low, _ = pca_different(hiddens_delay)

plt.figure()
plt.scatter(
    Low[:, 0],
    Low[:, 1],
    c=colors1,
    marker="o",
    s=50,  # 中等尺寸
    edgecolor="white",  # 白色边缘
    linewidths=0.8,  # 粗边缘线
    alpha=1,
    facecolors="none",
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
plt.figure()
plt.scatter(
    Low[:, 2],
    Low[:, 3],
    c=colors2,
    marker="o",
    s=50,  # 中等尺寸
    edgecolor="white",  # 白色边缘
    linewidths=0.8,  # 粗边缘线
    alpha=1,
    facecolors="none",
)
plt.xlabel("PC3")
plt.ylabel("PC4")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
plt.figure()
plt.scatter(
    Low[:, 4],
    Low[:, 5],
    c=colors3,
    marker="o",
    s=50,  # 中等尺寸
    edgecolor="white",  # 白色边缘
    linewidths=0.8,  # 粗边缘线
    alpha=1,
    facecolors="none",
)
plt.xlabel("PC6")
plt.ylabel("PC7")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
plt.figure()
# plt.scatter(Low[:,6],Low[:,7],colors=[x1],marker='*')
# plt.figure()
label = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]
explained_variance_ratio = pca_x.explained_variance_ratio_
print(explained_variance_ratio)
plt.bar(label, explained_variance_ratio[:8])

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", which="major", width=2, direction="in", length=4)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
plt.show()
