"""
### 方法

PCA分析等较为重复的部分在此处省略。

针对switch任务中训练得到的数据，前120条为forward任务（记为switch-forward），
后120条为backward任务（记为switch-backward）。分别使用顺序item1，2，3标记
switch-forward任务当中的数据，以及使用倒序item3，2，1标记switch-backward任务
当中的数据。在PCA分析结果中，若都能成功聚类（switch-forward PC1，2对应item1，类推；
switch-backward PC1，2对应item3，类推），则说明在switch任务中，rank是绝对的——
无论forward还是backward，网络在输出时都维持了一个绝对rank1，2，3，然后根据任务类型
选择插入item1，2，3还是3，2，1。

### 结果

- 针对switch-forward和switch-backward任务的post cue阶段（cue结束后，retrieval开始前）
的隐藏层神经元活动进行PCA分析，发现rank信息均依任务需求动态组织于三个子空间中。
- cue输入后，网络根据cue不同对网络状态进行灵活调整，逆序任务rank进行了翻转，subspace 1存储rank3，
subspace 3存储rank1，逆序任务仍旧按照顺序任务结构存储。

"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

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

hiddens = outputs_dict["hiddens"]
hiddens = np.tanh(hiddens)

hiddens_post_cue = hiddens[:, tps["post_cue_end"] - 10 : tps["post_cue_end"], :]


def compute_one(class_i):
    if class_i >= N_CLASS:
        # backward
        item_1, item_2, item_3 = PERMS[class_i - N_CLASS]
        return item_3, item_2, item_1
    else:
        # forward
        item_1, item_2, item_3 = PERMS[class_i]
        return item_1, item_2, item_3


# one-hot vector
X1 = []
X2 = []
X3 = []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)

base_colors, colors_rank1, colors_rank2, colors_rank3 = utils.get_color(X1, X2, X3)

Low_forward, var_explained_f, _ = utils.pca_single(hiddens_post_cue[0:N_CLASS, :, :])
Low_backward, var_explained_b, _ = utils.pca_single(hiddens_post_cue[N_CLASS:, :, :])

# ============= combined 2x3 grid using matplotlib =============
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# forward: first row
ax = axes[0, 0]
ax.scatter(Low_forward[:, 0], Low_forward[:, 1], c=colors_rank1[0:N_CLASS], marker="*")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Forward: PC1 vs PC2")

ax = axes[0, 1]
ax.scatter(Low_forward[:, 2], Low_forward[:, 3], c=colors_rank2[0:N_CLASS], marker="*")
ax.set_xlabel("PC3")
ax.set_ylabel("PC4")
ax.set_title("Forward: PC3 vs PC4")

ax = axes[0, 2]
ax.scatter(Low_forward[:, 4], Low_forward[:, 5], c=colors_rank3[0:N_CLASS], marker="*")
ax.set_xlabel("PC5")
ax.set_ylabel("PC6")
ax.set_title("Forward: PC5 vs PC6")

# backward: second row
ax = axes[1, 0]
ax.scatter(Low_backward[:, 0], Low_backward[:, 1], c=colors_rank1[N_CLASS:], marker="o")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Backward: PC1 vs PC2")

ax = axes[1, 1]
ax.scatter(Low_backward[:, 2], Low_backward[:, 3], c=colors_rank2[N_CLASS:], marker="o")
ax.set_xlabel("PC3")
ax.set_ylabel("PC4")
ax.set_title("Backward: PC3 vs PC4")

ax = axes[1, 2]
ax.scatter(Low_backward[:, 4], Low_backward[:, 5], c=colors_rank3[N_CLASS:], marker="o")
ax.set_xlabel("PC5")
ax.set_ylabel("PC6")
ax.set_title("Backward: PC5 vs PC6")


plt.tight_layout()
combined_path = FIG_DIR / "switch_post_cue_bothwards.png"
plt.savefig(combined_path)
print(f"Figure saved to {combined_path}")
plt.close()
