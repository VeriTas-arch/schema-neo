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

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

hiddens = outputs_dict["hiddens"]
hiddens = np.tanh(hiddens)


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


hiddens_out_1 = hiddens[:, tps["target1_start"] : tps["target1_end"], :]
hiddens_out_2 = hiddens[:, tps["target2_start"] : tps["target2_end"], :]
hiddens_out_3 = hiddens[:, tps["target3_start"] : tps["target3_end"], :]

bases_H1 = extract_subspace_bases(hiddens_out_1)
bases_H3 = extract_subspace_bases(hiddens_out_2)
bases_H5 = extract_subspace_bases(hiddens_out_3)


all_H = [hiddens_out_1, hiddens_out_2, hiddens_out_3]
all_bases = [bases_H1, bases_H3, bases_H5]
sub_names = ["sub1", "sub2", "sub3"]

# 构建 9×9 矩阵：三时间段 × 三子空间 之间的 CCA相似度
cca_matrix = np.zeros((9, 9))

for i in range(3):  # H1, H3, H5
    for si, sub_i in enumerate(sub_names):
        idx_i = i * 3 + si
        U = all_bases[i][sub_i]

        for j in range(3):  # H1, H3, H5
            for sj, sub_j in enumerate(sub_names):
                idx_j = j * 3 + sj
                V = all_bases[j][sub_j]

                cca_matrix[idx_i, idx_j] = cca_between_subspaces(
                    all_H[i], all_H[j], U, V
                )


labels = [
    "Out1_sub1",
    "Out1_sub2",
    "Out1_sub3",
    "Out2_sub1",
    "Out2_sub2",
    "Out2_sub3",
    "Out3_sub1",
    "Out3_sub2",
    "Out3_sub3",
]

plt.figure(figsize=(9, 7))
sns.heatmap(
    cca_matrix,
    cmap="viridis",
    annot=True,
    vmin=0,
    vmax=1,
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("CCA Similarity Between Subspaces During Retrieval Phase")
plt.tight_layout()
plt.savefig(FIG_DIR / "cca_subspace_similarity_heatmap.png")
print("Figure saved to:", FIG_DIR / "cca_subspace_similarity_heatmap.png")
plt.show()
