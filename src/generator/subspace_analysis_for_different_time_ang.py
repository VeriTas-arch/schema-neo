"""
随着stimulus输入, Win / Wout 与各个子空间的夹角变化。
> 结论的显著性存疑
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import generator.utils as utils

#### parameters
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name
params = utils.initialize_analysis(TASK, N_SAMPLES, FILENAME)

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]
STIM_DUR = params["STIM_DUR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

hiddens = outputs_dict["hiddens"]
Win = outputs_dict["Win"]
Wout = outputs_dict["Wout"]

hiddens = np.tanh(hiddens)

hidden_in_1 = hiddens[:, tps["stim1_start"] : tps["stim1_off"], :]
hidden_in_2 = hiddens[:, tps["stim2_start"] : tps["stim2_off"], :]
hidden_in_3 = hiddens[:, tps["stim3_start"] : tps["stim3_off"], :]

hidden_in_1_d = hiddens[:, tps["stim1_off"] : tps["stim1_end"], :]
hidden_in_2_d = hiddens[:, tps["stim2_off"] : tps["stim2_end"], :]
hidden_in_3_d = hiddens[:, tps["stim3_off"] : tps["stim3_end"], :]

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

hidden_out_1 = hiddens[:, tps["target1_start"] : tps["target1_end"], :]
hidden_out_2 = hiddens[:, tps["target2_start"] : tps["target2_end"], :]
hidden_out_3 = hiddens[:, tps["target3_start"] : tps["target3_end"], :]


def compute_one(class_i):
    if class_i >= N_CLASS:
        # backward
        item_1, item_2, item_3 = PERMS[class_i - N_CLASS]
        return item_3, item_2, item_1

    else:
        # forward
        item_1, item_2, item_3 = PERMS[class_i]
        return item_1, item_2, item_3


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


# 提取基底
bases_H1 = extract_subspace_bases(hidden_in_1)
bases_H3 = extract_subspace_bases(hidden_in_2)
bases_H5 = extract_subspace_bases(hidden_in_3)

ang_sub1 = [
    np.mean(utils.neo_compute_principal_angles(bases_H1["sub1"], Win.T)),
    np.mean(utils.neo_compute_principal_angles(bases_H3["sub1"], Win.T)),
    np.mean(utils.neo_compute_principal_angles(bases_H5["sub1"], Win.T)),
]

ang_sub2 = [
    np.mean(utils.neo_compute_principal_angles(bases_H1["sub2"], Win.T)),
    np.mean(utils.neo_compute_principal_angles(bases_H3["sub2"], Win.T)),
    np.mean(utils.neo_compute_principal_angles(bases_H5["sub2"], Win.T)),
]

ang_sub3 = [
    np.mean(utils.neo_compute_principal_angles(bases_H1["sub3"], Win.T)),
    np.mean(utils.neo_compute_principal_angles(bases_H3["sub3"], Win.T)),
    np.mean(utils.neo_compute_principal_angles(bases_H5["sub3"], Win.T)),
]


plt.figure(figsize=(10, 6))

# 绘图并添加标签
plt.plot(
    ["In1", "In2", "In3"],
    ang_sub1,
    label="Sub1",
    marker="o",
    linestyle="-",
    color="blue",
)
plt.plot(
    ["In1", "In2", "In3"],
    ang_sub2,
    label="Sub2",
    marker="s",
    linestyle="--",
    color="green",
)
plt.plot(
    ["In1", "In2", "In3"],
    ang_sub3,
    label="Sub3",
    marker="^",
    linestyle="-.",
    color="red",
)

plt.title("angle during input stage", fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=10)  # 自动选择最佳图例位置
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "input_time_ang.png")


bases_H1 = extract_subspace_bases(hidden_out_1)
bases_H3 = extract_subspace_bases(hidden_out_2)
bases_H5 = extract_subspace_bases(hidden_out_3)

ang_sub1 = [
    np.mean(utils.neo_compute_principal_angles(bases_H1["sub1"], Wout)),
    np.mean(utils.neo_compute_principal_angles(bases_H3["sub1"], Wout)),
    np.mean(utils.neo_compute_principal_angles(bases_H5["sub1"], Wout)),
]

ang_sub2 = [
    np.mean(utils.neo_compute_principal_angles(bases_H1["sub2"], Wout)),
    np.mean(utils.neo_compute_principal_angles(bases_H3["sub2"], Wout)),
    np.mean(utils.neo_compute_principal_angles(bases_H5["sub2"], Wout)),
]

ang_sub3 = [
    np.mean(utils.neo_compute_principal_angles(bases_H1["sub3"], Wout)),
    np.mean(utils.neo_compute_principal_angles(bases_H3["sub3"], Wout)),
    np.mean(utils.neo_compute_principal_angles(bases_H5["sub3"], Wout)),
]

# 假设 ang_sub1, ang_sub2, ang_sub3 均已定义
plt.figure(figsize=(10, 6))

# 绘图并添加标签
plt.plot(
    ["Out1", "Out2", "Out3"],
    ang_sub1,
    label="Sub1",
    marker="o",
    linestyle="-",
    color="blue",
)
plt.plot(
    ["Out1", "Out2", "Out3"],
    ang_sub2,
    label="Sub2",
    marker="s",
    linestyle="--",
    color="green",
)
plt.plot(
    ["Out1", "Out2", "Out3"],
    ang_sub3,
    label="Sub3",
    marker="^",
    linestyle="-.",
    color="red",
)

plt.grid(True, alpha=0.3)
plt.title("angle during output stage", fontsize=14, pad=20)
plt.legend(loc="best", fontsize=10)  # 自动选择最佳图例位置
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "output_time_ang.png")
