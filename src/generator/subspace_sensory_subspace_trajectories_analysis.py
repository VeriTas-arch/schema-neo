"""
This script has convergence problems when using Lasso.
Using Ridge regression instead, and it works.
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

    参数:
    hiddens: 3D神经活动矩阵 (trials, time, neurons)
    labels: 试验标签数组
    time_window_indices: 时间窗口索引 [start, end)
    alpha: Lasso正则化强度
    n_pca_components: PCA降维后的成分数

    返回:
    beta_matrix: 组合后的beta矩阵 (neurons, features)
    """
    # 提取指定时间窗口的数据
    window_data = hiddens[
        :, time_window_indices[0] : time_window_indices[1], :
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


def compute_time_segmented_betas(hiddens, labels, window_size=4):
    """
    分段计算时间窗口的beta矩阵

    参数:
    hiddens: 神经活动数据 (trials, time, neurons)
    labels: 试验标签
    window_size: 时间窗口大小

    返回:
    all_betas: 按时间组合的beta矩阵列表 [beta1, beta2, ...]
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


def plot_3d_trajectories(h, fig_path, mode="subplots"):
    """
    绘制三维轨迹图，可选分子图或合并图。

    参数:
    h : numpy数组, 形状为 [time, label1, label2, subplots]
    fig_path : 保存路径
    mode : "subplots"（分子图）或 "combined"（合并图）
    """
    assert h.shape[3] == 3, "最后一个维度应为3个子图"
    subplot_colors = ["b", "r", "y"]
    markers = ["^", "s", "o"]

    if mode == "subplots":
        fig = plt.figure(figsize=(6, 18))
        for subplot_idx in range(3):
            ax = fig.add_subplot(3, 1, subplot_idx + 1, projection="3d")
            ax.view_init(elev=30, azim=45 + subplot_idx * 60)
            color = subplot_colors[subplot_idx]
            for label in range(h.shape[1]):
                x_values = h[:, label, 0, subplot_idx]
                z_values = h[:, label, 1, subplot_idx]
                y_values = np.arange(h.shape[0])
                ax.plot(
                    x_values, y_values, z_values, color=color, linewidth=1.8, alpha=0.8
                )
                ax.scatter(
                    x_values[0],
                    y_values[0],
                    z_values[0],
                    color=color,
                    marker=markers[subplot_idx],
                    s=50,
                    edgecolor="k",
                )
                ax.scatter(
                    x_values[-1],
                    y_values[-1],
                    z_values[-1],
                    color=color,
                    marker=markers[subplot_idx],
                    s=50,
                    edgecolor="k",
                )
            ax.set_xlabel("X Value", fontsize=9, labelpad=8)
            ax.set_ylabel("Time Step", fontsize=9, labelpad=8)
            ax.set_zlabel("Z Value", fontsize=9, labelpad=8)
            ax.set_title(f"Memory {subplot_idx + 1}", fontsize=12, pad=16)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_box_aspect((1, 3.5, 1))
        ax.set_proj_type("ortho")
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")
        plt.close(fig)
    elif mode == "combined":
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        plt.title("Combined 3D Trajectories", y=1.02, fontsize=14)
        for subplot_idx in range(3):
            color = subplot_colors[subplot_idx]
            for label in range(h.shape[1]):
                x_values = h[:, label, 0, subplot_idx]
                z_values = h[:, label, 1, subplot_idx]
                y_values = np.arange(h.shape[0])
                ax.plot(
                    x_values,
                    y_values,
                    z_values,
                    color=color,
                    linewidth=2 - subplot_idx * 0.3,
                    alpha=0.7,
                )
                ax.scatter(
                    x_values[0],
                    y_values[0],
                    z_values[0],
                    color=color,
                    marker=markers[subplot_idx],
                    s=60,
                    edgecolor="k",
                )
                ax.scatter(
                    x_values[-1],
                    y_values[-1],
                    z_values[-1],
                    color=color,
                    marker=markers[subplot_idx],
                    s=60,
                    edgecolor="k",
                    facecolors="none",
                )
        ax.set_xlabel("X Value", fontsize=10, labelpad=10)
        ax.set_ylabel("Time Step", fontsize=10, labelpad=10)
        ax.set_zlabel("Z Value", fontsize=10, labelpad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect((1, 3.5, 1))
        ax.set_proj_type("ortho")
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")
        plt.close(fig)
    else:
        raise ValueError("mode must be 'subplots' or 'combined'")


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
            # 构造3D顶点列表: (Time, PC1, PC2) -> 这里我们将 Time 映射为 X轴
            # 注意：在 plot_composite_figure 中，我们将映射为: X=Time, Y=PC1, Z=PC2
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

    参数:
    hiddens: 神经活动数据
    labels: 标签数据
    tps: 时间点字典
    save_dir: 保存目录

    返回:
    h_pca_combined: Combined分析的PCA数据
    h_pca_encoding: Encoding分析的PCA数据
    """
    # 检查是否已存在保存的文件
    combined_path = save_dir / "h_pca_combined.npy"
    encoding_path = save_dir / "h_pca_encoding.npy"

    if combined_path.exists() and encoding_path.exists() and not overwrite:
        print("Loading cached PCA data from npy files...")
        h_pca_combined = np.load(combined_path)
        h_pca_encoding = np.load(encoding_path)
        return h_pca_combined, h_pca_encoding

    print("Computing PCA data...")

    # Combined Analysis
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

    # 计算轨迹数据
    segmented_betas_stim = compute_time_segmented_betas(
        hiddens[:, 0 : tps["delay_end"], :], labels, window_size=2
    )
    combined_beta_traj = np.stack(segmented_betas_stim, axis=1)

    windows_num_combined = int((tps["delay_end"]) / 2)

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

    # Encoding Analysis
    segmented_betas_enc = compute_time_segmented_betas(
        hiddens[:, 0 : tps["stim3_end"], :], labels, window_size=2
    )
    combined_beta_enc = np.stack(segmented_betas_enc, axis=1)
    beta_enc_basis = combined_beta_enc[:, -1, :]

    windows_num_enc = int((tps["stim3_end"] - 0) / 2)

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
    Layout: 4行1列
    Row 1: Combined analysis (所有3个记忆项)
    Row 2: Encoding Memory 1 (Cyan)
    Row 3: Encoding Memory 2 (Magenta)
    Row 4: Encoding Memory 3 (Orange/Yellow)
    """
    # 颜色定义 - 与参考代码一致
    color_r_hex = ["#00bfbf", "#ee82ee", "#daa520"]  # Cyan, Magenta, Orange
    colors = [
        tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)) for c in color_r_hex
    ]

    # Item 颜色（灰色用于共同表示）
    gray = tuple(i / 255 for i in [182, 182, 182])

    fig = plt.figure(figsize=(12, 10))

    time_len_enc = h_encoding.shape[0]
    time_steps_enc = np.arange(time_len_enc)

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

    azim_angle = -105  # 向x轴负方向移动
    elev_angle = 10  # 向下移动（降低约7度）

    # 计算x轴的box_aspect比例（基于时间长度）
    time_len_combined = h_combined.shape[0]
    time_len_encoding = h_encoding.shape[0]
    # 使用combined作为基准，计算encoding相对比例
    x_aspect_combined = 8.0
    x_aspect_encoding = x_aspect_combined * (time_len_encoding / time_len_combined)

    # 单独保存每张子图
    base_path = fig_path.parent / fig_path.stem
    subplot_names = ["combined", "memory1", "memory2", "memory3"]

    for idx, name in enumerate(subplot_names):
        # 创建单独的图形
        fig_single = plt.figure(figsize=(10, 3))
        ax_single = fig_single.add_subplot(111, projection="3d")

        # 设置视角和样
        ax_single.view_init(elev=elev_angle, azim=azim_angle)

        # 根据子图类型设置不同的box_aspect和x轴范围
        if idx == 0:
            ax_single.set_box_aspect((x_aspect_encoding, 1.5, 1.5))
            ax_single.set_xlim(0, time_len_combined - 1)
        else:
            ax_single.set_box_aspect((x_aspect_encoding, 1.5, 1.5))
            ax_single.set_xlim(0, time_len_combined - 1)

        ax_single.grid(False)

        # 设置背景透明
        ax_single.xaxis.pane.fill = False
        ax_single.yaxis.pane.fill = False
        ax_single.zaxis.pane.fill = False
        ax_single.xaxis.pane.set_edgecolor("w")
        ax_single.yaxis.pane.set_edgecolor("w")
        ax_single.zaxis.pane.set_edgecolor("w")

        # 重绘数据
        if idx == 0:
            # Combined subplot
            ax_single.set_ylim(y_min_top, y_max_top)
            ax_single.set_zlim(z_min_top, z_max_top)
            # ax_single.set_yticks([int(y_min_top), int(y_max_top)])
            # ax_single.set_zticks([int(z_min_top), int(z_max_top)])

            # 先收集所有三个stimulus时间点的六边形数据
            hexagon_data = {
                "stim1": {"x": [], "z": [], "time_idx": tps["stim1_off"] // 2 - 1},
                "stim2": {"x": [], "z": [], "time_idx": tps["stim2_off"] // 2 - 1},
                "stim3": {"x": [], "z": [], "time_idx": tps["stim3_off"] // 2 - 1},
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
                    # ax_single.scatter(
                    #     time_steps[-1], pc1[-1], pc2[-1], color=colors_list[label], s=15
                    # )

                    # 收集每个时间点的数据用于绘制六边形
                    hexagon_data["stim1"]["x"].append(pc1[tps["stim1_off"] // 2 - 1])
                    hexagon_data["stim1"]["z"].append(pc2[tps["stim1_off"] // 2 - 1])

                    hexagon_data["stim2"]["x"].append(pc1[tps["stim2_off"] // 2 - 1])
                    hexagon_data["stim2"]["z"].append(pc2[tps["stim2_off"] // 2 - 1])

                    hexagon_data["stim3"]["x"].append(pc1[tps["stim3_off"] // 2 - 1])
                    hexagon_data["stim3"]["z"].append(pc2[tps["stim3_off"] // 2 - 1])

            # 为每个stimulus时间点绘制六边形（使用相应的颜色）
            stim_names = ["stim1", "stim2", "stim3"]
            for i, stim_name in enumerate(stim_names):
                x_vals = np.array(hexagon_data[stim_name]["x"])
                z_vals = np.array(hexagon_data[stim_name]["z"])
                time_idx = hexagon_data[stim_name]["time_idx"]

                # 绘制六边形
                draw_end_polygon(
                    ax_single, x_vals, z_vals, time_idx, colors[i], alpha=0.3
                )

                # 在六边形中心绘制灰色散点
                st_time_idx = tps[f"{stim_name}_start"] // 2 - 1
                ax_single.scatter(
                    time_steps_enc[st_time_idx],
                    pc1[st_time_idx],
                    pc2[st_time_idx],
                    color=gray,
                    s=15,
                )

                # 只在六边形的顶点处绘制散点
                points_2d = np.column_stack((x_vals, z_vals))
                hull = ConvexHull(points_2d)
                # 只为ConvexHull的顶点绘制散点
                for vertex_idx in hull.vertices:
                    label_idx = vertex_idx % 6  # 循环使用6个颜色
                    ax_single.scatter(
                        time_idx,
                        x_vals[vertex_idx],
                        z_vals[vertex_idx],
                        color=colors_list[label_idx],
                        s=15,
                    )
        else:
            # Memory subplots
            ax_single.set_ylim(y_min_enc, y_max_enc)
            ax_single.set_zlim(z_min_enc, z_max_enc)
            # ax_single.set_yticks([int(y_min_enc), int(y_max_enc)])
            # ax_single.set_zticks([int(z_min_enc), int(z_max_enc)])

            c = colors[idx - 1]
            data_block = h_encoding[:, :, :, idx - 1]
            time_steps_enc = np.arange(h_encoding.shape[0])

            hexagon_x_list = []
            hexagon_z_list = []

            time_idx = tps[f"stim{idx}_start"] // 2 - 1

            for label in range(data_block.shape[1]):
                pc1 = data_block[:, label, 0]
                pc2 = data_block[:, label, 1]
                ax_single.plot(
                    time_steps_enc, pc1, pc2, color=c, linewidth=1.5, alpha=0.9
                )
                ax_single.scatter(
                    time_steps_enc[time_idx],
                    pc1[time_idx],
                    pc2[time_idx],
                    color=gray,
                    s=15,
                )

                ax_single.scatter(
                    time_steps_enc[-1], pc1[-1], pc2[-1], color=colors_list[label], s=15
                )

                hexagon_x_list.append(pc1[-1])
                hexagon_z_list.append(pc2[-1])

            # 绘制末端多边形
            draw_end_polygon(
                ax_single,
                np.array(hexagon_x_list),
                np.array(hexagon_z_list),
                time_steps_enc[-1],
                c,
                alpha=0.3,
            )

        # 设置x轴标签
        if idx == 3:
            ax_single.set_xticks(
                [
                    tps["stim1_start"] // 2,
                    tps["stim2_start"] // 2,
                    tps["stim3_start"] // 2,
                    tps["delay_start"] // 2,
                ],
                ["S1", "S2", "S3", "Delay"],
            )
            ax_single.set_xlabel("Time", fontsize=10, labelpad=20)
        else:
            ax_single.set_xticks([])
            ax_single.set_xlabel("")

        ax_single.set_yticks([])
        ax_single.set_zticks([])
        ax_single.set_ylabel("")
        ax_single.set_zlabel("")

        # 设置轴标签
        # ax_single.set_xlabel("Time", fontsize=10, labelpad=10)
        # ax_single.set_ylabel("rPC2", fontsize=10, labelpad=10)
        # ax_single.set_zlabel("rPC1", fontsize=10, labelpad=10)

        # 设置tick参数
        # ax_single.tick_params(axis="x", direction="out", length=8, pad=5)
        # ax_single.tick_params(axis="y", direction="out", length=4, pad=5)
        # ax_single.tick_params(axis="z", direction="out", length=4, pad=5)

        # 保存单独的子图
        single_fig_path = f"{base_path}_{name}.png"
        plt.savefig(single_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved subplot '{name}' to {single_fig_path}")
        plt.close(fig_single)

    plt.close(fig)


if __name__ == "__main__":
    r"""原始的绘图代码
    #     # 对整个时间序列进行分段计算
    #     segmented_betas = compute_time_segmented_betas(
    #         hiddens[:, 0 : tps["delay_end"], :], labels, window_size=2
    #     )

    #     # 各时间段的beta矩阵组合
    #     combined_beta = np.stack(segmented_betas, axis=1)  # (neurons, n_windows, 18)
    #     beta = combined_beta[:, -1, :]

    #     windows_num = int(tps["delay_end"] / 2)  # since window_size=2

    #     h_pca = np.zeros([windows_num, 6, n_components_ev, 3])
    #     for j in range(windows_num):
    #         for i in [1, 2, 3]:
    #             bt = combined_beta[:, j, :]
    #             analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
    #             pca_EV.fit(analy)
    #             h_pca[j, :, :, i - 1] = pca_EV.transform(
    #                 np.transpose(bt[:, (i - 1) * 6 : i * 6])
    #             )

    #     # 图1
    #     plot_3d_trajectories(
    #         h_pca, FIG_DIR / f"{TASK}_subspace_analysis_3d_corrected.png", mode="subplots"
    #     )

    #     delay_beta1 = compute_beta_for_time_window(
    #         hiddens, labels, [tps["stim1_start"], tps["stim1_off"]], alpha=0.001
    #     )
    #     delay_beta2 = compute_beta_for_time_window(
    #         hiddens, labels, [tps["stim2_start"], tps["stim2_off"]], alpha=0.001
    #     )
    #     delay_beta3 = compute_beta_for_time_window(
    #         hiddens, labels, [tps["stim3_start"], tps["stim3_off"]], alpha=0.001
    #     )

    #     # 各时间段的beta矩阵组合
    #     combined_beta = (
    #         delay_beta1[:, 0:6] + delay_beta2[:, 6:12] + delay_beta3[:, 12:]
    #     ) / 3  # (neurons, 6)

    #     segmented_betas = compute_time_segmented_betas(
    #         hiddens[:, tps["stim1_start"] : tps["delay_end"], :], labels, window_size=2
    #     )

    #     # 各时间段的beta矩阵组合
    #     combined_beta2 = np.stack(segmented_betas, axis=1)  # (neurons, n_windows, 18)

    #     windows_num = int(
    #         (tps["delay_end"] - tps["stim1_start"]) / 2
    #     )  # since window_size=2

    #     h_pca = np.zeros([windows_num, 6, n_components_ev, 3])
    #     analy = np.transpose(combined_beta[:, :])

    #     pca_EV.fit(analy)
    #     for j in range(windows_num):
    #         for i in [1, 2, 3]:
    #             bt = combined_beta2[:, j, :]

    #             h_pca[j, :, :, i - 1] = pca_EV.transform(
    #                 np.transpose(bt[:, (i - 1) * 6 : i * 6])
    #             )

    #     # 图2
    #     plot_3d_trajectories(
    #         h_pca, FIG_DIR / f"{TASK}_subspace_analysis_3d_combined.png", mode="combined"
    #     )

    #     # 由于图1当中轨迹的张开主要由delay阶段的变化贡献
    #     # 因此在这里额外计算stimulus输入阶段的memory子空间轨迹
    #     segmented_betas = compute_time_segmented_betas(
    #         hiddens[:, 0 : tps["stim3_end"], :], labels, window_size=2
    #     )

    #     # 各时间段的beta矩阵组合
    #     combined_beta = np.stack(segmented_betas, axis=1)  # (neurons, n_windows, 18)
    #     beta = combined_beta[:, -1, :]

    #     windows_num = int((tps["stim3_end"] - 0) / 2)  # since window_size=2

    #     h_pca = np.zeros([windows_num, 6, n_components_ev, 3])
    #     for j in range(windows_num):
    #         for i in [1, 2, 3]:
    #             bt = combined_beta[:, j, :]
    #             analy = np.transpose(beta[:, (i - 1) * 6 : i * 6])
    #             pca_EV.fit(analy)
    #             h_pca[j, :, :, i - 1] = pca_EV.transform(
    #                 np.transpose(bt[:, (i - 1) * 6 : i * 6])
    #             )

    #     # 图3
    #     plot_3d_trajectories(
    #         h_pca, FIG_DIR / f"{TASK}_subspace_analysis_3d_encoding.png", mode="subplots"
    #     )
    """

    # ---------------------------------------------------------
    # 计算或加载PCA数据
    # ---------------------------------------------------------
    h_pca_combined, h_pca_encoding = compute_pca_data(
        hiddens, labels, tps, FIG_DIR, overwrite=False
    )

    # ---------------------------------------------------------
    # 生成最终组合图
    # ---------------------------------------------------------
    plot_composite_figure(
        h_pca_combined,
        h_pca_encoding,
        FIG_DIR / f"{TASK}_subspace_analysis_composite.png",
    )
