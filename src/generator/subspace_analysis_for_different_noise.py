"""
应该是正常的。
复现结束。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import generator.utils as utils

#### parameters
TASK = "forward"
N_SAMPLES = 120
FILENAME = Path(__file__).name

params_0 = utils.initialize_analysis(
    TASK, N_SAMPLES, FILENAME, single_stage_noise=dict(stage="stim3", noise_level=0.0)
)
params_1 = utils.initialize_analysis(
    TASK, N_SAMPLES, FILENAME, single_stage_noise=dict(stage="stim3", noise_level=2.0)
)

FIG_DIR = params_0["FIG_DIR"]
RNN_CONFIG = params_0["RNN_CONFIG"]
LOAD_MODEL_DIR = params_0["LOAD_MODEL_DIR"]

tps = params_0["TIMEPOINTS"]

val_loader_0 = params_0["VAL_LOADER"]
val_loader_1 = params_1["VAL_LOADER"]

var = np.zeros([50, 3])
scaler = StandardScaler()


# 将数据A和B分别投影到主成分空间
def project_to_pca(data):
    data_scaled = scaler.transform(data)
    return pca.transform(data_scaled)


def calculate_group_variance(pc_scores, group):
    scores_group = pc_scores[:, group]  # (n_trials, 2)
    covariance = np.cov(scores_group, rowvar=False)  # (2,2)
    return np.trace(covariance)  # 等价于方差之和


for i in tqdm(range(50), desc="noisy samples", colour="red"):
    outputs_dict_0 = utils.get_rnn_outputs(
        RNN_CONFIG, LOAD_MODEL_DIR, val_loader_0, framework="jax"
    )
    hiddens_0 = outputs_dict_0["hiddens"]
    hiddens_0 = np.tanh(hiddens_0)

    outputs_dict_1 = utils.get_rnn_outputs(
        RNN_CONFIG, LOAD_MODEL_DIR, val_loader_1, framework="jax"
    )
    hiddens_1 = outputs_dict_1["hiddens"]
    hiddens_1 = np.tanh(hiddens_1)

    in_0 = np.mean(hiddens_0[:, tps["stim3_start"] : tps["stim3_off"], :], axis=1)
    in_1 = np.mean(hiddens_1[:, tps["stim3_start"] : tps["stim3_off"], :], axis=1)

    # 合并数据A和B用于标准化和PCA训练
    data_combined = np.concatenate([in_0, in_1], axis=0)
    data_combined_scaled = scaler.fit_transform(data_combined)

    # 保留全部或足够的主成分（例如前10个）
    pca = PCA(n_components=10)
    pca.fit(data_combined_scaled)

    pc_scores_A = project_to_pca(in_0)  # shape=(n_trials, n_pcs)
    pc_scores_B = project_to_pca(in_1)
    GROUP1 = [0, 1]  # PC1+PC2
    GROUP2 = [2, 3]  # PC3+PC4
    GROUP3 = [4, 5]  # PC5+PC6

    # 数据A的方差
    var_A_g1 = calculate_group_variance(pc_scores_A, GROUP1)  # scalar
    var_A_g2 = calculate_group_variance(pc_scores_A, GROUP2)
    var_A_g3 = calculate_group_variance(pc_scores_A, GROUP3)

    # 数据B的方差
    var_B_g1 = calculate_group_variance(pc_scores_B, GROUP1)
    var_B_g2 = calculate_group_variance(pc_scores_B, GROUP2)
    var_B_g3 = calculate_group_variance(pc_scores_B, GROUP3)

    delta_var_g1 = (var_B_g1 - var_A_g1) / var_A_g1  # PC1+PC2的变化比例
    delta_var_g2 = (var_B_g2 - var_A_g2) / var_A_g2  # PC3+PC4的变化比例
    delta_var_g3 = (var_B_g3 - var_A_g3) / var_A_g3
    var[i, 0] = delta_var_g1
    var[i, 1] = delta_var_g2
    var[i, 2] = delta_var_g3

# labels = ["Sub1", "Sub2"]
# delta_vars = np.mean(var, axis=0)[0:2] * 100  # 点估计值
# std = np.std(var, axis=0)[0:2]

# plt.figure(figsize=(6, 6))
# bars = plt.bar(labels, delta_vars)
# # plt.show()
# plt.savefig(FIG_DIR / "sub12_with_noise_stim2.png")

labels = ["Sub1", "Sub2", "Sub3"]
delta_vars = np.mean(var, axis=0) * 100  # 点估计值
std = np.std(var, axis=0)

plt.figure(figsize=(6, 6))
bars = plt.bar(labels, delta_vars)
# plt.show()
plt.savefig(FIG_DIR / "sub123_with_noise_stim3.png")

print(delta_vars)
