"""
动力学投影分析：
识别在 encoding 阶段对子空间旋转贡献最大的神经元
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import generator.utils as utils

# ============================================================
# parameters
# ============================================================
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name
params_base = utils.initialize_analysis_legacy(TASK, N_SAMPLES, FILENAME)
params_silence_whh = utils.initialize_analysis_legacy(
    TASK, N_SAMPLES, FILENAME, silence_whh=True
)

N_HID = params_base["N_HID"]
N_CLASS = params_base["N_CLASS"]
PERMS = params_base["PERMS"]
FIG_DIR = params_base["FIG_DIR"]

outputs_dict = params_base["OUTPUTS_DICT"]
tps = params_base["TIMEPOINTS"]

# ============================================================
# load hidden states
# shape: (trial, time, neuron)
# ============================================================
hiddens_normal = outputs_dict["hiddens"]
Win = outputs_dict["Win"]
Wout = outputs_dict["Wout"]

hiddens_normal = np.tanh(hiddens_normal)

outputs_dict_silence = params_silence_whh["OUTPUTS_DICT"]
hiddens_silence = outputs_dict_silence["hiddens"]


X = np.zeros([N_SAMPLES, 18])
X1 = []
X2 = []
X3 = []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = PERMS[i % N_CLASS]
    X[i, x1] = 1
    X[i, x2 + 6] = 1
    X[i, x3 + 12] = 1
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)


def get_output_windows(hiddens, tps):
    hiddens_output1 = hiddens[:, tps["target1_start"] : tps["target1_end"], :]
    hiddens_output2 = hiddens[:, tps["target2_start"] : tps["target2_end"], :]
    hiddens_output3 = hiddens[:, tps["target3_start"] : tps["target3_end"], :]

    return hiddens_output1, hiddens_output2, hiddens_output3


def compute_output_accuracy(outputs, x):
    # 对每个trial取时间维度上的最大值
    output_max = np.max(outputs, axis=1)  # shape: (n_trials, n_classes)

    # 获取预测标签（最大值的索引）
    pred_labels = np.argmax(output_max, axis=1)

    # 计算准确率
    acc = np.sum(pred_labels == np.array(x)) / len(x)

    return acc


def compute_all_stage_accuracy(outputs_dict, tps):
    outputs = outputs_dict["outputs"]  # shape: (n_trials, time, n_classes)

    # 计算各输出时间段的准确率
    acc1 = compute_output_accuracy(
        outputs[:, tps["target1_start"] : tps["target1_end"], :], X1
    )
    acc2 = compute_output_accuracy(
        outputs[:, tps["target2_start"] : tps["target2_end"], :], X2
    )
    acc3 = compute_output_accuracy(
        outputs[:, tps["target3_start"] : tps["target3_end"], :], X3
    )

    acc_mean = (acc1 + acc2 + acc3) / 3.0

    return acc1, acc2, acc3, acc_mean


# ============================================================
# plot functions
# ============================================================
def plot_win_angles(Win, hiddens, name_suffix=""):
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

    hidden_in_1 = hiddens[:, tps["stim1_start"] : tps["stim1_off"], :]
    hidden_in_2 = hiddens[:, tps["stim2_start"] : tps["stim2_off"], :]
    hidden_in_3 = hiddens[:, tps["stim3_start"] : tps["stim3_off"], :]

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
    plt.savefig(FIG_DIR / f"input_time_ang_{name_suffix}.png")
    print(f"Saved figure: input_time_ang_{name_suffix}.png")
    plt.close()


plot_win_angles(Win, hiddens_normal, name_suffix="normal")
plot_win_angles(Win, hiddens_silence, name_suffix="silence")

# ============================================================
# 计算并打印模型输出准确率
# ============================================================
print("\n" + "=" * 50)
print("计算模型输出准确率")
print("=" * 50)

# 计算正常模型的准确率
acc1_normal, acc2_normal, acc3_normal, acc_mean_normal = compute_all_stage_accuracy(
    outputs_dict, tps
)
print("\n正常模型准确率:")
print(f"  Output1: {acc1_normal:.4f}")
print(f"  Output2: {acc2_normal:.4f}")
print(f"  Output3: {acc3_normal:.4f}")
print(f"  平均值: {acc_mean_normal:.4f}")

# 计算silence模型的准确率
acc1_silence, acc2_silence, acc3_silence, acc_mean_silence = compute_all_stage_accuracy(
    outputs_dict_silence, tps
)
print("\nSilence模型准确率:")
print(f"  Output1: {acc1_silence:.4f}")
print(f"  Output2: {acc2_silence:.4f}")
print(f"  Output3: {acc3_silence:.4f}")
print(f"  平均值: {acc_mean_silence:.4f}")
print("=" * 50 + "\n")
