"""
分析forward任务中的encoding阶段, 不同时期同角标子空间之间的关系。
例如, 希望证明stimulus 1输入时的subspace 1与stimulus 2 输入时的subspace 2仅为旋转关系。
计算不同阶段的相同角标子空间之间的旋转矩阵, 然后旋转数据, 再投影到对应PCA上。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import itertools

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

hiddens = outputs_dict["hiddens"]
labels = outputs_dict["labels"]

hiddens = np.tanh(hiddens)


def compute_one(class_i: int) -> tuple[int, int, int]:
    """
    Return the three items for a given permutation index.

    Parameters
    ----------
    class_i : int
        Index into the global ``PERMS`` permutation list.

    Returns
    -------
    tuple of int
        (item_1, item_2, item_3) corresponding to ranks 1..3.
    """
    item_1, item_2, item_3 = PERMS[class_i]
    return item_1, item_2, item_3


# 构建线性回归模型
# one-hot vector
X1, X2, X3 = [], [], []
for i in np.arange(0, N_SAMPLES):
    x1, x2, x3 = compute_one(i)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)

base_colors, colors_rank1, colors_rank2, colors_rank3 = utils.get_color(X1, X2, X3)

hiddens_input1 = hiddens[:, tps["stim1_start"] : tps["stim1_off"], :]
hiddens_input2 = hiddens[:, tps["stim2_start"] : tps["stim2_off"], :]
hiddens_input3 = hiddens[:, tps["stim3_start"] : tps["stim3_off"], :]

hiddens_input1_delay = hiddens[:, tps["stim1_off"] : tps["stim1_end"], :]
hiddens_input2_delay = hiddens[:, tps["stim2_off"] : tps["stim2_end"], :]
hiddens_input3_delay = hiddens[:, tps["stim3_off"] : tps["stim3_end"], :]

hiddens_stim1 = hiddens[:, tps["stim1_start"] : tps["stim1_end"], :]
hiddens_stim2 = hiddens[:, tps["stim2_start"] : tps["stim2_end"], :]
hiddens_stim3 = hiddens[:, tps["stim3_start"] : tps["stim3_end"], :]

hiddens_delay = hiddens[:, tps["delay_start"] : tps["delay_end"], :]

INPUT1 = np.mean(hiddens_input1, axis=1).reshape(-1, N_HID)
INPUT2 = np.mean(hiddens_input2, axis=1).reshape(-1, N_HID)
INPUT3 = np.mean(hiddens_input3, axis=1).reshape(-1, N_HID)

INPUT1_DELAY = np.mean(hiddens_input1_delay, axis=1).reshape(-1, N_HID)
INPUT2_DELAY = np.mean(hiddens_input2_delay, axis=1).reshape(-1, N_HID)
INPUT3_DELAY = np.mean(hiddens_input3_delay, axis=1).reshape(-1, N_HID)

DELAY = np.mean(hiddens_delay, axis=1).reshape(-1, N_HID)


def plot_cross_angles(angle_matrix: np.ndarray, labels: list[str]) -> None:
    """
    Plot a heatmap of pairwise mean principal angles between subspaces.

    Parameters
    ----------
    angle_matrix : ndarray
        Pairwise matrix of mean principal angles (degrees).
    labels : list of str
        Tick labels for rows/columns of the heatmap.
    """
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
    plt.savefig(
        FIG_DIR / "cross_subspace_principal_angles.png", bbox_inches="tight", dpi=150
    )
    plt.close()


def plot_stage_pca_grid(
    Lows: list[np.ndarray],
    var_explaineds: list[np.ndarray],
    colors: list[list],
    row_labels: list[str],
    fig_prefix: str = "three_stage",
    xlim: tuple[float, float] = (-7, 7),
    ylim: tuple[float, float] = (-7, 7),
) -> Path:
    """
    Plot a 3x4 grid showing PCA projections (PC1-2, PC3-4, PC5-6) and variance.

    Parameters
    ----------
    Lows : list of ndarray
        List of per-class low-dim arrays, one per row/phase.
    var_explaineds : list of ndarray
        Corresponding explained variance arrays for each PCA.
    colors : list of list
        Color arrays used for plotting each phase.
    row_labels : list of str
        Titles for each row.
    fig_prefix : str, optional
        Output file prefix.
    xlim, ylim : tuple, optional
        Axis limits for scatter plots.
    """

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    row_num = len(Lows)

    for i in range(row_num):
        Low = Lows[i]
        var_exp = var_explaineds[i]

        # PC1-2
        ax = axes[i, 0]
        ax.scatter(Low[:, 0], Low[:, 1], c=colors[0], marker="o", alpha=0.8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{row_labels[i]}: PC1-2")

        # PC3-4
        ax = axes[i, 1]
        ax.scatter(Low[:, 2], Low[:, 3], c=colors[1], marker="o", alpha=0.8)
        ax.set_xlabel("PC3")
        ax.set_ylabel("PC4")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{row_labels[i]}: PC3-4")

        # PC5-6
        ax = axes[i, 2]
        ax.scatter(Low[:, 4], Low[:, 5], c=colors[2], marker="o", alpha=0.8)
        ax.set_xlabel("PC5")
        ax.set_ylabel("PC6")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{row_labels[i]}: PC5-6")

        # var explained bar
        ax = axes[i, 3]
        ax.bar(np.arange(1, len(var_exp) + 1), var_exp, color="#6C8EB4")
        ax.set_xticks(np.arange(1, len(var_exp) + 1))
        ax.set_xlabel("PC")
        ax.set_ylabel("Explained variance")
        ax.set_ylim([0, max(0.6, float(var_exp.max()) + 0.05)])
        ax.set_title(f"{row_labels[i]}: Explained Variance")

    plt.tight_layout()
    out_path = FIG_DIR / f"{fig_prefix}_grid.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print("Saved 3x4 grid to:", out_path)
    plt.close()


def align_2d_signs_by_centroids(
    L_ref: np.ndarray,
    L_tgt: np.ndarray,
    labels_ref: np.ndarray,
    labels_tgt: np.ndarray,
    pc_pair: tuple[int, int] = (0, 1),
    min_count: int = 1,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    Choose sign flips on a 2D target embedding to minimize centroid distance.

    For a given pair of principal component indices (pc_pair) this function
    tries the four combinations of sign flips on the two target coordinates
    and selects the one that minimizes the average squared distance between
    class centroids in the 2D subspace. Classes with fewer than ``min_count``
    samples in either reference or target are ignored.

    Parameters
    ----------
    L_ref : ndarray, shape (n_samples, n_dims)
        Reference low-dimensional embedding (source of desired orientation).
    L_tgt : ndarray, shape (n_samples, n_dims)
        Target low-dimensional embedding to be sign-flipped and aligned.
    labels_ref : ndarray, shape (n_samples,)
        Class labels corresponding to ``L_ref`` samples.
    labels_tgt : ndarray, shape (n_samples,)
        Class labels corresponding to ``L_tgt`` samples.
    pc_pair : tuple of int, optional
        Two indices of principal components to compare (default ``(0, 1)``).
    min_count : int, optional
        Minimum number of samples per class required in both reference and
        target to include that class in centroid computations (default 1).

    Returns
    -------
    L_tgt_aligned : ndarray
        Copy of ``L_tgt`` with the selected sign flips applied on the two
        coordinates specified by ``pc_pair``.
    best_signs : tuple
        Chosen sign combination (s0, s1), each either 1 or -1.
    best_score : float
        Average squared centroid distance for the chosen signs; ``np.inf`` if
        no class met ``min_count``.
    """

    p0, p1 = pc_pair
    A: np.ndarray = L_ref[:, [p0, p1]]
    B: np.ndarray = L_tgt[:, [p0, p1]]
    classes: np.ndarray = np.unique(labels_ref)

    # precompute centroids and counts for each class once
    cent_a = {}
    cent_b = {}
    counts = {}
    for c in classes:
        idx_a = np.where(labels_ref == c)[0]
        idx_b = np.where(labels_tgt == c)[0]
        counts[c] = (len(idx_a), len(idx_b))
        if len(idx_a) > 0:
            cent_a[c] = A[idx_a].mean(axis=0)
        if len(idx_b) > 0:
            cent_b[c] = B[idx_b].mean(axis=0)

    best_score: float = np.inf
    best_signs: tuple[int, int] = (1, 1)
    best_B: np.ndarray = B.copy()

    # try 4 sign combos; use precomputed centroids to compute score cheaply
    for s0, s1 in itertools.product([1, -1], [1, -1]):
        total: float = 0.0
        count: int = 0
        for c in classes:
            na, nb = counts[c]
            if na < min_count or nb < min_count:
                continue
            ca = cent_a[c]
            cb = cent_b[c] * np.array([s0, s1])  # apply sign flip to centroid coords
            total += np.sum((ca - cb) ** 2)
            count += 1
        score = total / count if count > 0 else np.inf
        if score < best_score:
            best_score = score
            best_signs = (s0, s1)
            # apply chosen signs to B once
            best_B = B.copy()
            best_B[:, 0] *= s0
            best_B[:, 1] *= s1

    L_new: np.ndarray = L_tgt.copy()
    L_new[:, p0] = best_B[:, 0]
    L_new[:, p1] = best_B[:, 1]
    return L_new, best_signs, best_score


def visualize_rotation_relation(
    LowA: np.ndarray,
    LowB: np.ndarray,
    colorsA: list,
    colorsB: list,
    labelA: str,
    labelB: str,
    pc_slice: tuple[int, int] = (0, 1),
    fig_name: str = "rotation_alignment.png",
) -> tuple[Path, np.ndarray, float, float]:
    """
    Compute and visualize a 2D rotation aligning two PC subspaces (Kabsch + centroid
    correction) and save the alignment figures.

    The function performs a Kabsch (Procrustes) alignment on the two selected
    PC dimensions, computes a centroid-based mean-angle correction, applies the
    total rotation, saves diagnostic scatter plots (original, Kabsch-rotated,
    and corrected), and returns the corrected figure path and rotation metrics.

    Parameters
    ----------
    LowA : ndarray, shape (n_samples, >=max(pc_slice)+1)
        Reference low-dimensional data (target for alignment).
    LowB : ndarray, shape (n_samples, >=max(pc_slice)+1)
        Source low-dimensional data to be rotated into LowA's frame.
    colorsA : sequence
        Colors for samples in LowA (used for plotting/grouping).
    colorsB : sequence
        Colors for samples in LowB (used for plotting/grouping).
    labelA : str
        Label used in plot legend for LowA.
    labelB : str
        Label used in plot legend for LowB.
    pc_slice : tuple of int, optional
        Two principal component indices to compare (default ``(0, 1)``).
    fig_name : str, optional
        Filename for saved figure (default ``"rotation_alignment.png"``).

    Returns
    -------
    out_path_corr : pathlib.Path
        File path to the saved corrected-alignment figure ("*_corrected.png").
    R_total : ndarray, shape (2, 2)
        Total 2x2 rotation matrix applied (Kabsch followed by centroid correction).
    angle_total_deg : float
        Rotation angle in degrees corresponding to ``R_total``.
    recon_err_corr : float
        Relative reconstruction error after applying the corrected rotation.
    """
    p0, p1 = pc_slice
    A: np.ndarray = LowA[:, [p0, p1]]
    B: np.ndarray = LowB[:, [p0, p1]]

    # Center the point clouds (important for Kabsch/Procrustes)
    A_mean: np.ndarray = A.mean(axis=0)
    B_mean: np.ndarray = B.mean(axis=0)
    A_c: np.ndarray = A - A_mean
    B_c: np.ndarray = B - B_mean

    # Kabsch: compute H = B_c^T A_c, then SVD(H) = U S Vt, R = V U^T
    H: np.ndarray = B_c.T @ A_c
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    U, S, Vt = np.linalg.svd(H)
    R: np.ndarray = Vt.T @ U.T

    # enforce proper rotation (no reflection) if det(R) < 0
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # apply rotation to centered B and translate to A mean
    B_rot: np.ndarray = (B_c @ R) + A_mean

    # reconstruction error (relative) for Kabsch-only
    recon_err: float = np.linalg.norm(A - B_rot) / (np.linalg.norm(A) + 1e-12)

    # diagnostics for Kabsch
    print("Kabsch singular values:", S)
    print("det(R):", np.linalg.det(R))

    # rotation angle (in degrees) for 2D orthogonal matrix (Kabsch)
    angle_deg: float = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))

    # save original kabsch alignment figure
    plt.figure(figsize=(6, 6))
    plt.scatter(A[:, 0], A[:, 1], c=colorsA, alpha=0.7, label=f"{labelA} (raw)")
    plt.scatter(B[:, 0], B[:, 1], c=colorsB, alpha=0.3, label=f"{labelB} (raw)")
    plt.scatter(
        B_rot[:, 0],
        B_rot[:, 1],
        c=colorsB,
        marker="x",
        alpha=0.9,
        label=f"{labelB} (rotated)",
    )
    plt.xlabel(f"PC{p0+1}")
    plt.ylabel(f"PC{p1+1}")
    plt.title(
        f"Rotation Alignment (Kabsch): {labelA} vs {labelB} (PC{p0+1}-PC{p1+1})\nangle={angle_deg:.2f}°, recon_err={recon_err:.3e}"
    )
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    out_path = FIG_DIR / fig_name
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    # --- centroid-angle based correction ---
    # use colorsA to group samples by class/color
    colorsA_arr: np.ndarray = np.array(colorsA)
    unique_colors: np.ndarray = np.unique(colorsA_arr)

    angle_diffs: list[float] = []
    for c in unique_colors:
        idx = np.where(colorsA_arr == c)[0]
        if len(idx) == 0:
            continue
        ca: np.ndarray = A[idx].mean(axis=0)
        cb_rot: np.ndarray = B_rot[idx].mean(axis=0)
        vec_a: np.ndarray = ca - A_mean
        vec_b: np.ndarray = cb_rot - A_mean
        norm_a: float = np.linalg.norm(vec_a)
        norm_b: float = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            # skip degenerate
            continue
        ang_a = np.arctan2(vec_a[1], vec_a[0])
        ang_b = np.arctan2(vec_b[1], vec_b[0])
        diff = ang_b - ang_a
        # wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        angle_diffs.append(diff)

    mean_sin: float = float(np.mean(np.sin(angle_diffs)))
    mean_cos: float = float(np.mean(np.cos(angle_diffs)))
    mean_diff: float = np.arctan2(mean_sin, mean_cos)

    print(f"Centroid mean angle correction (deg): {np.rad2deg(mean_diff):.4f}")

    # build correction rotation matrix Rc (2x2) that rotates by mean_diff
    c: float = np.cos(mean_diff)
    s: float = np.sin(mean_diff)
    Rc: np.ndarray = np.array([[c, -s], [s, c]])

    # total rotation: first apply original R, then correction Rc => R_total = R @ Rc
    R_total: np.ndarray = R @ Rc

    # apply total rotation to centered B and translate to A mean
    B_rot_corr: np.ndarray = (B_c @ R_total) + A_mean

    # reconstruction error (relative) after correction
    recon_err_corr: float = np.linalg.norm(A - B_rot_corr) / (np.linalg.norm(A) + 1e-12)

    # rotation angle for total
    angle_total_deg: float = np.rad2deg(np.arctan2(R_total[1, 0], R_total[0, 0]))

    # save corrected alignment figure
    plt.figure(figsize=(6, 6))
    plt.scatter(A[:, 0], A[:, 1], c=colorsA, alpha=0.7, label=f"{labelA} (raw)")
    plt.scatter(
        B_rot_corr[:, 0],
        B_rot_corr[:, 1],
        c=colorsB,
        marker="+",
        alpha=0.9,
        label=f"{labelB} (kabsch+corr)",
    )

    #### debug plots for comparison
    # plt.scatter(B[:, 0], B[:, 1], c=colorsB, alpha=0.2, label=f"{labelB} (raw)")
    # plt.scatter(
    #     B_rot[:, 0],
    #     B_rot[:, 1],
    #     c=colorsB,
    #     marker="x",
    #     alpha=0.5,
    #     label=f"{labelB} (kabsch)",
    # )

    plt.xlabel(f"PC{p0+1}")
    plt.ylabel(f"PC{p1+1}")
    plt.title(
        f"Rotation Alignment: {labelA} vs {labelB} (PC{p0+1}-PC{p1+1})\nangle={angle_total_deg:.2f}°, recon_err={recon_err_corr:.3e}"
    )
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    out_path_corr: Path = FIG_DIR / fig_name.replace(".png", "_corrected.png")
    plt.savefig(out_path_corr, bbox_inches="tight", dpi=150)
    plt.close()

    return out_path_corr, R_total, angle_total_deg, recon_err_corr


if __name__ == "__main__":
    layers = {
        "Input1": INPUT1,
        "Input1_D": INPUT1_DELAY,
        "Input2": INPUT2,
        "Input2_D": INPUT2_DELAY,
        "Input3": INPUT3,
        "Input3_D": INPUT3_DELAY,
    }

    subspace_bases, angle_matrix, labels = utils.compute_subspace_angles(layers)
    plot_cross_angles(angle_matrix, labels)

    # compute Low and var_explained for three input windows
    Low1, var1, _ = utils.pca_single(hiddens_stim1)
    Low2, var2, _ = utils.pca_single(hiddens_stim2)
    Low3, var3, _ = utils.pca_single(hiddens_stim3)

    plot_stage_pca_grid(
        [Low1, Low2, Low3],
        [var1, var2, var3],
        [colors_rank1, colors_rank2, colors_rank3],
        ["Input1", "Input2", "Input3"],
        fig_prefix="input",
    )

    # align Low2/Low3 to Low1 by sign flips in PC1-2 and PC3-4 using class centroids
    labels1 = np.array(X1)
    labels2 = np.array(X2)
    labels3 = np.array(X3)

    # pc12, backward alignment: align Low1 to Low2, then Low2 to Low3
    Low12_a_pc12, signs12_pc12, score12_pc12 = align_2d_signs_by_centroids(
        Low2, Low1, labels2, labels1, pc_pair=(0, 1), min_count=1
    )
    Low23_a_pc12, signs23_pc12, score23_pc12 = align_2d_signs_by_centroids(
        Low3, Low2, labels3, labels2, pc_pair=(0, 1), min_count=1
    )

    # pc34, backward alignment: align Low2 to Low3
    Low23_a_pc34, signs23_pc34, score23_pc34 = align_2d_signs_by_centroids(
        Low3, Low2, labels3, labels2, pc_pair=(2, 3), min_count=1
    )

    # visualize and save rotation alignments using backward projection: 1->2 and 2->3
    # Project Low1 into Low2 (target Low2, source Low1)
    out1, R12, angle12, err12 = visualize_rotation_relation(
        Low2,
        Low12_a_pc12,
        colors_rank1,
        colors_rank1,
        "Input2",
        "Input1",
        pc_slice=(0, 1),
        fig_name="rotation_input1_input2_PC12.png",
    )
    print("Saved rotation alignment (Low1 -> Low2, PC1-2):", out1)
    print("R12=\n", R12)
    print(f"angle12={angle12:.3f} deg, recon_err={err12:.3e}")

    # Project Low2 into Low3 (target Low3, source Low2)
    out2, R23, angle23, err23 = visualize_rotation_relation(
        Low3,
        Low23_a_pc12,
        colors_rank1,
        colors_rank1,
        "Input3",
        "Input2",
        pc_slice=(0, 1),
        fig_name="rotation_input2_input3_PC12.png",
    )
    print("Saved rotation alignment (Low2 -> Low3, PC1-2):", out2)
    print("R23=\n", R23)
    print(f"angle23={angle23:.3f} deg, recon_err={err23:.3e}")

    # also check PC3-4 between Low2 and Low3 (project Low2 into Low3)
    out3, R23_34, angle23_34, err23_34 = visualize_rotation_relation(
        Low3,
        Low23_a_pc34,
        colors_rank2,
        colors_rank2,
        "Input3",
        "Input2",
        pc_slice=(2, 3),
        fig_name="rotation_input2_input3_PC34.png",
    )
    print("Saved rotation alignment (Low2 -> Low3, PC3-4):", out3)
    print("R23_34=\n", R23_34)
    print(f"angle23_34={angle23_34:.3f} deg, recon_err={err23_34:.3e}")
