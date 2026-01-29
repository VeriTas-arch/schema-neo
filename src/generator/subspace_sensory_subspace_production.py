"""
This script has convergence problems when using Lasso.
Using Ridge regression instead, and it works.
Window size set to 5 for all trajectory calculations.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import generator.utils as utils

# 设置全局字体为Arial
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 显示负号

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

labels = outputs_dict["labels"]
hiddens = outputs_dict["hiddens"]
hiddens = np.tanh(hiddens)

labels = labels.reshape(-1)


print(tps)

def compute_one(class_i):
    class_i = class_i % N_CLASS
    item_1, item_2, item_3 = PERMS[class_i]
    return item_1, item_2, item_3


## pca analysis, initialize a pca_EV template
n_components_ev = 6
pca_EV = PCA(n_components=n_components_ev)

colors_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]


def compute_beta_for_time_window(
    hiddens, labels, time_window_indices, alpha=0.001, n_pca_components=8
):
    """
    计算指定时间窗口的beta系数矩阵
    """
    # 提取指定时间窗口的数据
    start, end = time_window_indices
    # 防止切片为空
    if start >= end:
        return np.zeros((hiddens.shape[2], 18))

    window_data = hiddens[
        :, start : end, :
    ]  # (trials, window_size, neurons)
    window_mean = np.mean(window_data, axis=1)  # (trials, neurons)

    N_trials = window_data.shape[0]
    X = np.zeros((N_trials, 18))

    for trial in range(N_trials):
        item1, item2, item3 = compute_one(labels[trial])
        X[trial, item1] = 1
        X[trial, item2 + 6] = 1
        X[trial, item3 + 12] = 1

    N_HID = window_data.shape[2]  # 神经元数量
    beta = np.zeros((N_HID, 18))  # 每个神经元对应18个特征

    model = Ridge(alpha=alpha)
    for neuron in range(N_HID):
        model.fit(X, window_mean[:, neuron])
        beta[neuron, :] = model.coef_

    return beta


def compute_time_segmented_betas(hiddens, labels, window_size=5):
    """
    分段计算时间窗口的beta矩阵
    """
    n_time = hiddens.shape[1]
    n_windows = n_time // window_size

    all_betas = []
    for seg in range(n_windows):
        start = seg * window_size
        end = start + window_size

        # 计算当前时间段的beta
        beta_seg = compute_beta_for_time_window(
            hiddens, labels, [start, end], alpha=0.001
        )
        all_betas.append(beta_seg)

    return all_betas


def draw_end_polygon(ax, x_vals, z_vals, time_val, color, alpha=0.3):
    """
    辅助函数：在轨迹末端绘制多边形面 (使用ConvexHull确定顶点顺序)
    """
    points_2d = np.column_stack((x_vals, z_vals))
    if len(points_2d) >= 3:
        try:
            hull = ConvexHull(points_2d)
            # 按Hull的顺序排列顶点
            x_ordered = points_2d[hull.vertices, 0]
            z_ordered = points_2d[hull.vertices, 1]
            verts = [list(zip([time_val] * len(x_ordered), x_ordered, z_ordered))]

            poly = Poly3DCollection(
                verts, alpha=alpha, facecolor=color, edgecolor=color
            )
            ax.add_collection3d(poly)
        except Exception:
            pass  # 如果点共线或无法形成Hull，则跳过


def compute_pca_data(hiddens, labels, tps, save_dir, overwrite=False):
    """
    计算并保存PCA数据到npy文件
    修改：全程使用 window_size = 5
    """
    combined_path = save_dir / "h_pca_combined_w5.npy"
    encoding_path = save_dir / "h_pca_encoding_w5.npy"

    if combined_path.exists() and encoding_path.exists() and not overwrite:
        print("Loading cached PCA data from npy files...")
        h_pca_combined = np.load(combined_path)
        h_pca_encoding = np.load(encoding_path)
        return h_pca_combined, h_pca_encoding

    print("Computing PCA data with window size 5...")
    WINDOW_SIZE = 5

    # Combined Analysis Basis
    delay_beta1 = compute_beta_for_time_window(
        hiddens, labels, [tps["stim1_start"], tps["stim1_off"]], alpha=0.001
    )
    delay_beta2 = compute_beta_for_time_window(
        hiddens, labels, [tps["stim2_start"], tps["stim2_off"]], alpha=0.001
    )
    delay_beta3 = compute_beta_for_time_window(
        hiddens, labels, [tps["stim3_start"], tps["stim3_off"]], alpha=0.001
    )

    # 组合定义子空间
    beta_combined_basis = (
        delay_beta1[:, 0:6] + delay_beta2[:, 6:12] + delay_beta3[:, 12:]
    ) / 3

    # ==========================================
    # Combined Trajectory Calculation (Fixed Window = 5)
    # ==========================================
    segmented_betas_stim = compute_time_segmented_betas(
        hiddens[:, 0 : tps["delay_end"], :], labels, window_size=WINDOW_SIZE
    )
    combined_beta_traj = np.stack(segmented_betas_stim, axis=1)

    windows_num_combined = len(segmented_betas_stim)

    # 初始化存储 Combined 数据的数组
    h_pca_combined = np.zeros([windows_num_combined, 6, n_components_ev, 3])

    # PCA Fit
    analy = np.transpose(beta_combined_basis[:, :])
    pca_EV.fit(analy)

    # PCA Transform
    for j in range(windows_num_combined):
        for i in [1, 2, 3]:
            bt = combined_beta_traj[:, j, :]
            h_pca_combined[j, :, :, i - 1] = pca_EV.transform(
                np.transpose(bt[:, (i - 1) * 6 : i * 6])
            )

    # ==========================================
    # Encoding Trajectory Calculation (Fixed Window = 5)
    # ==========================================
    segmented_betas_enc = compute_time_segmented_betas(
        hiddens[:, 0 : tps["delay_end"], :], labels, window_size=WINDOW_SIZE
    )
    combined_beta_enc = np.stack(segmented_betas_enc, axis=1)

    # Encoding Basis (基于最后时刻)
    beta_enc_basis = combined_beta_enc[:, -1, :]

    windows_num_enc = len(segmented_betas_enc)

    # 初始化存储 Encoding 数据的数组
    h_pca_encoding = np.zeros([windows_num_enc, 6, n_components_ev, 3])

    for j in range(windows_num_enc):
        for i in [1, 2, 3]:
            bt = combined_beta_enc[:, j, :]
            analy = np.transpose(beta_enc_basis[:, (i - 1) * 6 : i * 6])
            pca_EV.fit(analy)
            h_pca_encoding[j, :, :, i - 1] = pca_EV.transform(
                np.transpose(bt[:, (i - 1) * 6 : i * 6])
            )

    # 保存到npy文件
    np.save(combined_path, h_pca_combined)
    np.save(encoding_path, h_pca_encoding)
    print(f"Saved PCA data to {combined_path} and {encoding_path}")

    return h_pca_combined, h_pca_encoding


def plot_composite_figure(h_combined, h_encoding, fig_path):
    """
    绘制符合参考代码风格的组合图。
    适应固定窗口(size=5)的索引计算。
    """
    # 颜色定义
    color_r_hex = ["#00bfbf", "#ee82ee", "#daa520"]  # Cyan, Magenta, Orange
    colors = [
        tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)) for c in color_r_hex
    ]
    gray = tuple(i / 255 for i in [182, 182, 182])

    WINDOW_SIZE = 5

    fig = plt.figure(figsize=(12, 10))

    time_len_enc = h_encoding.shape[0]
    time_steps_enc = np.arange(time_len_enc)

    # 索引计算辅助函数
    def get_time_idx(t):
        """将绝对时间转换为绘图数组的索引"""
        return int(t / WINDOW_SIZE)

    def get_end_idx(t):
        """获取结束于时间t的窗口索引 (通常用于取状态值)"""
        return get_time_idx(t) - 1

    # 第一张子图（Combined）的范围
    y_min_top, y_max_top = float("inf"), float("-inf")
    z_min_top, z_max_top = float("inf"), float("-inf")

    for subplot_idx in range(3):
        data_block = h_combined[:, :, :, subplot_idx]
        y_min_top = min(y_min_top, data_block[:, :, 0].min())
        y_max_top = max(y_max_top, data_block[:, :, 0].max())
        z_min_top = min(z_min_top, data_block[:, :, 1].min())
        z_max_top = max(z_max_top, data_block[:, :, 1].max())

    # 下方子图（Encoding）的统一范围
    y_min_enc, y_max_enc = float("inf"), float("-inf")
    z_min_enc, z_max_enc = float("inf"), float("-inf")

    for i in range(3):
        data_block = h_encoding[:, :, :, i]
        y_min_enc = min(y_min_enc, data_block[:, :, 0].min())
        y_max_enc = max(y_max_enc, data_block[:, :, 0].max())
        z_min_enc = min(z_min_enc, data_block[:, :, 1].min())
        z_max_enc = max(z_max_enc, data_block[:, :, 1].max())

    azim_angle = -105
    elev_angle = 10

    time_len_combined = h_combined.shape[0]

    # Box aspect ratio calculation
    # 调整比例，因为点数变少了 (120/5 = 24 points vs 120/2 = 60 points)
    # 保持物理长度比例一致
    x_aspect_combined = 8.0
    x_aspect_encoding = x_aspect_combined * (time_len_enc / time_len_combined)

    base_path = fig_path.parent / fig_path.stem
    subplot_names = ["combined", "memory1", "memory2", "memory3"]

    for idx, name in enumerate(subplot_names):
        fig_single = plt.figure(figsize=(10, 3))
        ax_single = fig_single.add_subplot(111, projection="3d")
        ax_single.view_init(elev=elev_angle, azim=azim_angle)

        if idx == 0:
            ax_single.set_box_aspect((x_aspect_encoding, 1.5, 1.5))
            # ax_single.set_xlim(0, time_len_combined - 1)
        else:
            ax_single.set_box_aspect((x_aspect_encoding, 1.5, 1.5))
            # ax_single.set_xlim(0, time_len_combined - 1)

        ax_single.grid(False)
        ax_single.xaxis.pane.fill = False
        ax_single.yaxis.pane.fill = False
        ax_single.zaxis.pane.fill = False
        ax_single.xaxis.pane.set_edgecolor("w")
        ax_single.yaxis.pane.set_edgecolor("w")
        ax_single.zaxis.pane.set_edgecolor("w")

        if idx == 0:
            # Combined subplot
            ax_single.set_ylim(y_min_top, y_max_top)
            ax_single.set_zlim(z_min_top, z_max_top)

            idx_s1_off = get_end_idx(tps["stim1_off"])
            idx_s2_off = get_end_idx(tps["stim2_off"])
            idx_s3_off = get_end_idx(tps["stim3_off"])

            hexagon_data = {
                "stim1": {"x": [], "z": [], "time_idx": idx_s1_off},
                "stim2": {"x": [], "z": [], "time_idx": idx_s2_off},
                "stim3": {"x": [], "z": [], "time_idx": idx_s3_off},
            }

            for subplot_idx in range(3):
                c = colors[subplot_idx]
                data_block = h_combined[:, :, :, subplot_idx]
                time_steps = np.arange(h_combined.shape[0])

                for label in range(data_block.shape[1]):
                    pc1 = data_block[:, label, 0]
                    pc2 = data_block[:, label, 1]
                    ax_single.plot(
                        time_steps, pc1, pc2, color=c, linewidth=1.5, alpha=0.8
                    )
                    ax_single.scatter(
                        time_steps[0], pc1[0], pc2[0], color=gray, s=15, alpha=0.5
                    )

                    hexagon_data["stim1"]["x"].append(pc1[idx_s1_off])
                    hexagon_data["stim1"]["z"].append(pc2[idx_s1_off])

                    hexagon_data["stim2"]["x"].append(pc1[idx_s2_off])
                    hexagon_data["stim2"]["z"].append(pc2[idx_s2_off])

                    hexagon_data["stim3"]["x"].append(pc1[idx_s3_off])
                    hexagon_data["stim3"]["z"].append(pc2[idx_s3_off])

            stim_names = ["stim1", "stim2", "stim3"]
            for i, stim_name in enumerate(stim_names):
                x_vals = np.array(hexagon_data[stim_name]["x"])
                z_vals = np.array(hexagon_data[stim_name]["z"])
                time_idx = hexagon_data[stim_name]["time_idx"]

                draw_end_polygon(
                    ax_single, x_vals, z_vals, time_idx, colors[i], alpha=0.3
                )

                points_2d = np.column_stack((x_vals, z_vals))
                hull = ConvexHull(points_2d)
                for vertex_idx in hull.vertices:
                    label_idx = vertex_idx % 6
                    ax_single.scatter(
                        time_idx,
                        x_vals[vertex_idx],
                        z_vals[vertex_idx],
                        color=colors_list[label_idx],
                        s=15,
                    )
        else:
            # Memory subplots
            c = colors[idx - 1]
            data_block = h_encoding[:, :, :, idx - 1]

            # 使用特定Memory的Start时间
            time_idx_start = get_time_idx(tps[f"stim{idx}_start"])

            hexagon_x_list = []
            hexagon_z_list = []

            for label in range(data_block.shape[1]):
                pc1 = data_block[:, label, 0]
                pc2 = data_block[:, label, 1]
                ax_single.plot(
                    time_steps_enc, pc1, pc2, color=c, linewidth=1.5, alpha=0.9
                )

                # 绘制Start点
                if time_idx_start < len(pc1):
                    ax_single.scatter(
                        time_steps_enc[time_idx_start],
                        pc1[time_idx_start],
                        pc2[time_idx_start],
                        color=gray,
                        s=15,
                    )

                # End点
                ax_single.scatter(
                    time_steps_enc[-1], pc1[-1], pc2[-1], color=colors_list[label], s=15
                )

                hexagon_x_list.append(pc1[-1])
                hexagon_z_list.append(pc2[-1])

            draw_end_polygon(
                ax_single,
                np.array(hexagon_x_list),
                np.array(hexagon_z_list),
                time_steps_enc[-1],
                c,
                alpha=0.3,
            )

        # 获取 S1, S2, S3 的开始索引
        idx_s1 = get_time_idx(tps["stim1_start"])
        idx_s2 = get_time_idx(tps["stim2_start"])
        idx_s3 = get_time_idx(tps["stim3_start"])

        tick_indices = [
            idx_s1,
            idx_s2,
            idx_s3,
        ]

        ax_single.set_xticks(
            tick_indices,
            ["S1", "S2", "S3"],
        )
        ax_single.set_xlabel("Time", fontsize=10, labelpad=20)

        ax_single.set_yticks([])
        ax_single.set_zticks([])
        ax_single.set_ylabel("")
        ax_single.set_zlabel("")

        single_fig_path = f"{base_path}_{name}.png"
        plt.savefig(single_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved subplot '{name}' to {single_fig_path}")
        plt.close(fig_single)

    plt.close(fig)

if __name__ == "__main__": # 注意这里的引号修复
    # 1. 计算原始数据
    h_pca_combined, h_pca_encoding = compute_pca_data(
        hiddens, labels, tps, FIG_DIR, overwrite=False
    )
    WINDOW_SIZE = 5

    print("\n" + "="*40)
    print(f"Window Count Analysis (Window Size = {WINDOW_SIZE})")
    print("="*40)

    # 定义要查看的刺激阶段
    stages = ["stim1", "stim2", "stim3"]

    for stage in stages:
        # 获取原始时间点
        # 注意：你的字典里用的键名是 'stimX_start' 和 'stimX_off'
        t_start = tps[f"{stage}_start"]
        t_end = tps[f"{stage}_end"]

        # 计算对应的窗口索引 (向下取整)
        w_idx_start = int(t_start / WINDOW_SIZE)
        w_idx_end = int(t_end / WINDOW_SIZE)

        # 计算个数
        count = w_idx_end - w_idx_start

        print(f"{stage.upper()}:")
        print(f"  Raw Time : {t_start} -> {t_end} (Duration: {t_end - t_start})")
        print(f"  Win Index: {w_idx_start} -> {w_idx_end}")
        print(f"  Count    : {count} windows")
        print("-" * 40)

    # =======================================================
    print("Applying extension to the last 3 time windows of encoding data...")
    if h_pca_encoding.shape[0] >= 3:
        head_part = h_pca_encoding[:-1]
        tail_part = h_pca_encoding[-1:]
        extended_tail = np.repeat(tail_part, 2, axis=0)
        h_pca_encoding_modified = np.concatenate([head_part, extended_tail], axis=0)
        print(f"Original shape: {h_pca_encoding.shape}")
        print(f"Modified shape: {h_pca_encoding_modified.shape}")
    else:
        h_pca_encoding_modified = h_pca_encoding


    # =======================================================
    # 修改结束
    # =======================================================

    # 2. 绘图
    plot_composite_figure(
        h_pca_combined, h_pca_encoding_modified, FIG_DIR / "pca_composite_w5.png"
    )
