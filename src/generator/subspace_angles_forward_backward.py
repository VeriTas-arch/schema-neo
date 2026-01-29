"""
绘制switch任务中前向和后向子空间表示的主角度热力图
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import generator.utils as utils

#### parameters
TASK = "switch"
N_SAMPLES = 240
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

hiddens_pre_cue = hiddens[:, tps["pre_cue_start"] : tps["cue_start"], :]
hiddens_cue = hiddens[:, tps["cue_start"] : tps["cue_end"], :]
hiddens_post_cue = hiddens[:, tps["cue_end"] : tps["post_cue_end"], :]

pre_cue_forward = hiddens_pre_cue[0:N_CLASS]
pre_cue_backward = hiddens_pre_cue[N_CLASS:]

cue_forward = hiddens_cue[0:N_CLASS]
cue_backward = hiddens_cue[N_CLASS:]

post_cue_forward = hiddens_post_cue[0:N_CLASS]
post_cue_backward = hiddens_post_cue[N_CLASS:]

PRE_CUE_FORWARD = np.mean(pre_cue_forward, axis=1).reshape(-1, N_HID)
PRE_CUE_BACKWARD = np.mean(pre_cue_backward, axis=1).reshape(-1, N_HID)

CUE_FORWARD = np.mean(cue_forward, axis=1).reshape(-1, N_HID)
CUE_BACKWARD = np.mean(cue_backward, axis=1).reshape(-1, N_HID)

POST_CUE_FORWARD = np.mean(post_cue_forward, axis=1).reshape(-1, N_HID)
POST_CUE_BACKWARD = np.mean(post_cue_backward, axis=1).reshape(-1, N_HID)


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
    plt.savefig(FIG_DIR / "cross_subspace_principal_angles.png")


layers = {
    "pre_cue_forward": PRE_CUE_FORWARD,
    "pre_cue_backward": PRE_CUE_BACKWARD,
    "cue_forward": CUE_FORWARD,
    "cue_backward": CUE_BACKWARD,
    "post_cue_forward": POST_CUE_FORWARD,
    "post_cue_backward": POST_CUE_BACKWARD,
}

subspace_bases, angle_matrix, labels = utils.compute_subspace_angles(layers)


plot_cross_angles(angle_matrix, labels)
