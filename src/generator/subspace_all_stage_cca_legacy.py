import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

import generator.utils as utils
from sklearn.cross_decomposition import CCA
import seaborn as sns
from sklearn.decomposition import PCA

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

plt.rcParams["font.sans-serif"] = "Arial"


def cca_between_subspaces(H1, H2, U, V):
    """
    计算两个 2D 子空间之间的 CCA 相似度。

    参数:
        H1: 时间段1 的隐藏状态 (n_trials, time, n_neurons)
        H2: 时间段2 的隐藏状态 (n_trials, time, n_neurons)
        U: 时间段1 的 2D 子空间基 (n_neurons, 2)
        V: 时间段2 的 2D 子空间基 (n_neurons, 2)

    返回:
        cca_corr: 两个 canonical correlation 的平均值
    """

    # 时间平均
    X1 = H1.mean(axis=1)  # shape [n_trials, n_neurons]
    X2 = H2.mean(axis=1)

    # 投影到子空间
    Z1 = X1 @ U  # shape [n_trials, 2]
    Z2 = X2 @ V  # shape [n_trials, 2]

    # 运行 CCA
    cca = CCA(n_components=2)
    Z1_c, Z2_c = cca.fit_transform(Z1, Z2)

    # 计算两个 canonical dims 的相关系数
    c1 = np.corrcoef(Z1_c[:, 0], Z2_c[:, 0])[0, 1]
    c2 = np.corrcoef(Z1_c[:, 1], Z2_c[:, 1])[0, 1]

    return np.mean([c1, c2])


def compute_cca_matrix_single_model(params):
    """
    为单个模型计算 CCA 相似度矩阵

    参数:
        params: 包含模型参数和数据的字典

    返回:
        cca_matrix: 12×12 的 CCA 相似度矩阵
        subspace_info: 子空间信息字典
    """
    outputs_dict = params["OUTPUTS_DICT"]
    tps = params["TIMEPOINTS"]

    hiddens = outputs_dict["hiddens"]
    hiddens = np.tanh(hiddens)

    # 提取各时间段的隐藏状态
    hiddens_stim1 = hiddens[:, tps["stim1_start"] : tps["stim1_end"], :]
    hiddens_stim2 = hiddens[:, tps["stim2_start"] : tps["stim2_end"], :]
    hiddens_stim3 = hiddens[:, tps["stim3_start"] : tps["stim3_end"], :]
    hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]
    hiddens_out_1 = hiddens[:, tps["target1_start"] : tps["target1_end"], :]
    hiddens_out_2 = hiddens[:, tps["target2_start"] : tps["target2_end"], :]
    hiddens_out_3 = hiddens[:, tps["target3_start"] : tps["target3_end"], :]

    # 提取各时间段的子空间基底
    H_mean_stim1 = hiddens_stim1.mean(axis=1).reshape(-1, N_HID)
    pca_stim1 = PCA(n_components=2).fit(H_mean_stim1)
    stim1_bases = {"sub1": pca_stim1.components_[[0, 1]].T}

    H_mean_stim2 = hiddens_stim2.mean(axis=1).reshape(-1, N_HID)
    pca_stim2 = PCA(n_components=4).fit(H_mean_stim2)
    stim2_bases = {
        "sub1": pca_stim2.components_[[0, 1]].T,
        "sub2": pca_stim2.components_[[2, 3]].T,
    }

    H_mean_stim3 = hiddens_stim3.mean(axis=1).reshape(-1, N_HID)
    pca_stim3 = PCA(n_components=6).fit(H_mean_stim3)
    stim3_bases = {
        "sub1": pca_stim3.components_[[0, 1]].T,
        "sub2": pca_stim3.components_[[2, 3]].T,
        "sub3": pca_stim3.components_[[4, 5]].T,
    }

    H_mean_delay = hiddens_delay.mean(axis=1).reshape(-1, N_HID)
    pca_delay = PCA(n_components=6).fit(H_mean_delay)
    delay_bases = {
        "sub1": pca_delay.components_[[0, 1]].T,
        "sub2": pca_delay.components_[[2, 3]].T,
        "sub3": pca_delay.components_[[4, 5]].T,
    }

    H_mean_out1 = hiddens_out_1.mean(axis=1).reshape(-1, N_HID)
    pca_out1 = PCA(n_components=2).fit(H_mean_out1)
    out1_bases = {"sub1": pca_out1.components_[[0, 1]].T}

    H_mean_out2 = hiddens_out_2.mean(axis=1).reshape(-1, N_HID)
    pca_out2 = PCA(n_components=2).fit(H_mean_out2)
    out2_bases = {"sub1": pca_out2.components_[[0, 1]].T}

    H_mean_out3 = hiddens_out_3.mean(axis=1).reshape(-1, N_HID)
    pca_out3 = PCA(n_components=2).fit(H_mean_out3)
    out3_bases = {"sub1": pca_out3.components_[[0, 1]].T}

    # 定义子空间标签
    labels = [
        "Stim1_sub1",  # Stim1: PC1-2
        "Stim2_sub1",  # Stim2: PC1-2
        "Stim3_sub1",  # Stim3: PC1-2
        "Delay_sub1",  # Delay: PC1-2
        "Out1_sub1",  # Out1: PC1-2
        "Stim2_sub2",  # Stim2: PC3-4
        "Stim3_sub2",  # Stim3: PC3-4
        "Delay_sub2",  # Delay: PC3-4
        "Out2_sub1",  # Out2: PC1-2
        "Stim3_sub3",  # Stim3: PC5-6
        "Delay_sub3",  # Delay: PC5-6
        "Out3_sub1",  # Out3: PC1-2
    ]

    # 定义每个标签对应的 H 和 bases
    subspace_info = {
        "Stim1_sub1": {"H": hiddens_stim1, "U": stim1_bases["sub1"]},
        "Stim2_sub1": {"H": hiddens_stim2, "U": stim2_bases["sub1"]},
        "Stim2_sub2": {"H": hiddens_stim2, "U": stim2_bases["sub2"]},
        "Stim3_sub1": {"H": hiddens_stim3, "U": stim3_bases["sub1"]},
        "Stim3_sub2": {"H": hiddens_stim3, "U": stim3_bases["sub2"]},
        "Stim3_sub3": {"H": hiddens_stim3, "U": stim3_bases["sub3"]},
        "Delay_sub1": {"H": hiddens_delay, "U": delay_bases["sub1"]},
        "Delay_sub2": {"H": hiddens_delay, "U": delay_bases["sub2"]},
        "Delay_sub3": {"H": hiddens_delay, "U": delay_bases["sub3"]},
        "Out1_sub1": {"H": hiddens_out_1, "U": out1_bases["sub1"]},
        "Out2_sub1": {"H": hiddens_out_2, "U": out2_bases["sub1"]},
        "Out3_sub1": {"H": hiddens_out_3, "U": out3_bases["sub1"]},
    }

    # 构建 12×12 矩阵：12个子空间之间的 CCA相似度
    n_subspaces = len(labels)
    cca_matrix = np.zeros((n_subspaces, n_subspaces))

    for i in range(n_subspaces):
        for j in range(n_subspaces):
            label_i = labels[i]
            label_j = labels[j]
            H1 = subspace_info[label_i]["H"]
            U = subspace_info[label_i]["U"]
            H2 = subspace_info[label_j]["H"]
            V = subspace_info[label_j]["U"]

            cca_matrix[i, j] = cca_between_subspaces(H1, H2, U, V)

    return cca_matrix, labels


# 对所有模型计算 CCA 矩阵
print(f"处理 {len(params_list)} 个模型的 CCA 相似度...")
n_models = len(params_list)
all_cca_matrices = []

for model_idx, params in enumerate(params_list):
    print(f"  处理模型 {model_idx + 1}/{n_models}...")
    cca_matrix, labels = compute_cca_matrix_single_model(params)
    all_cca_matrices.append(cca_matrix)

# 计算所有模型的平均 CCA 矩阵
cca_matrix_mean = np.mean(all_cca_matrices, axis=0)
print("完成平均 CCA 矩阵计算")

# 定义标签映射字典：将代码标签映射到图注中使用的标签
label_mapping = {
    "Stim1_sub1": r"$S_1$-$\mathcal{U}_1(R_1)$",
    "Stim2_sub1": r"$S_2$-$\mathcal{U}_1(R_1)$",
    "Stim3_sub1": r"$S_3$-$\mathcal{U}_1(R_1)$",
    "Delay_sub1": r"$M$-$\mathcal{U}_1(R_1)$",
    "Out1_sub1": r"$O_1$-$\mathcal{U}_1(R_1)$",
    "Stim2_sub2": r"$S_2$-$\mathcal{U}_2(R_2)$",
    "Stim3_sub2": r"$S_3$-$\mathcal{U}_2(R_2)$",
    "Delay_sub2": r"$M$-$\mathcal{U}_2(R_2)$",
    "Out2_sub1": r"$O_2$-$\mathcal{U}_1(R_2)$",
    "Stim3_sub3": r"$S_3$-$\mathcal{U}_3(R_3)$",
    "Delay_sub3": r"$M$-$\mathcal{U}_3(R_3)$",
    "Out3_sub1": r"$O_3$-$\mathcal{U}_1(R_3)$",
}

# 使用映射字典转换标签为图注标签
display_labels = [label_mapping[label] for label in labels]

plt.figure(figsize=(9, 7))
sns.heatmap(
    cca_matrix_mean,
    cmap="viridis",
    annot=False,
    vmin=0,
    vmax=1,
    xticklabels=display_labels,
    yticklabels=display_labels,
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sfig_all_stage_subspace_cca_heatmap.pdf", bbox_inches='tight')
print(f"Figure saved to: {FIG_DIR}/sfig_all_stage_subspace_cca_heatmap.pdf")
plt.show()
