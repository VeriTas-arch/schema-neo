"""
绘制不同阶段不同子空间之间的主角度热力图
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

## dataset
hiddens = outputs_dict["hiddens"]
hiddens = np.tanh(hiddens)


def compute_one(class_i):
    if class_i >= N_CLASS:
        # backward
        item_1, item_2, item_3 = PERMS[class_i - N_CLASS]
        return item_3, item_2, item_1

    else:
        # forward
        item_1, item_2, item_3 = PERMS[class_i]
        return item_1, item_2, item_3


# 构建线性回归模型
# one-hot vector
X1 = []
X2 = []
X3 = []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)

hiddens_input1 = hiddens[:, tps["stim1_start"] : tps["stim1_off"], :]
hiddens_input2 = hiddens[:, tps["stim2_start"] : tps["stim2_off"], :]
hiddens_input3 = hiddens[:, tps["stim3_start"] : tps["stim3_off"], :]

hiddens_input1_delay = hiddens[:, tps["stim1_off"] : tps["stim1_end"], :]
hiddens_input2_delay = hiddens[:, tps["stim2_off"] : tps["stim2_end"], :]
hiddens_input3_delay = hiddens[:, tps["stim3_off"] : tps["stim3_end"], :]

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

INPUT1 = np.mean(hiddens_input1, axis=1).reshape(-1, N_HID)
INPUT2 = np.mean(hiddens_input2, axis=1).reshape(-1, N_HID)
INPUT3 = np.mean(hiddens_input3, axis=1).reshape(-1, N_HID)

INPUT1_DELAY = np.mean(hiddens_input1_delay, axis=1).reshape(-1, N_HID)
INPUT2_DELAY = np.mean(hiddens_input2_delay, axis=1).reshape(-1, N_HID)
INPUT3_DELAY = np.mean(hiddens_input3_delay, axis=1).reshape(-1, N_HID)

DELAY = np.mean(hiddens_delay, axis=1).reshape(-1, N_HID)


def plot_cross_angles(angle_matrix, labels):
    """绘制所有子空间间夹角热力图"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        angle_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Average Angle (Degrees)"},
    )
    plt.title("Cross-Subspace Principal Angles Between Layers")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FIG_DIR / "cross_subspace_principal_angles.png")


layers = {
    "Input1": INPUT1,
    "Input1_D": INPUT1_DELAY,
    "Input2": INPUT2,
    "Input2_D": INPUT2_DELAY,
    "Input3": INPUT3,
    "Input3_D": INPUT3_DELAY,
    # "Delay": DELAY,
}
subspace_bases, angle_matrix, labels = utils.compute_subspace_angles(layers)


plot_cross_angles(angle_matrix, labels)
