"""
分析对于forward任务的encoding阶段，对不同rank子空间的旋转均有显著贡献的神经元
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles
from typing import Tuple, Dict
import csv

import generator.utils as utils

#### parameters
base_cmap = plt.get_cmap("tab10")
base_colors = [base_cmap(i) for i in range(6)]

rank_factors = {1: 1.05}
shaded_colors = [utils.shade_color(base_colors[i], rank_factors[1]) for i in range(6)]


def components_from_window(
    hiddens_window: np.ndarray, n_components: int = 8
) -> np.ndarray:
    """
    Fit PCA on the provided window and return orthonormal component basis matrix V (n_hid x n_components).
    """
    _, _, pca_x = utils.pca_single(hiddens_window, n_components=n_components)
    # sklearn PCA.components_ has shape (n_components, n_features)
    V = pca_x.components_.T  # shape (n_hid, n_components)
    # ensure orthonormal columns (should be from PCA)
    return V


def rotation_metric_between_bases(V1: np.ndarray, V2: np.ndarray) -> float:
    """
    Given two orthonormal bases V1, V2 (shape n_hid x k), return a scalar rotation metric.
    We use principal angles via scipy.linalg.subspace_angles (returns radians).
    Metric = Frobenius norm of angle vector (i.e. sqrt(sum angles^2)).
    """
    # subspace_angles returns array of principal angles in radians
    angles = subspace_angles(V1, V2)  # length = min(k1,k2)
    return np.linalg.norm(angles)  # scalar


# Benjamini-Hochberg FDR
def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    pvals: 1D array
    returns boolean mask of which hypotheses are rejected at FDR alpha
    """
    p = np.asarray(pvals)
    n = p.size
    idx = np.argsort(p)
    sorted_p = p[idx]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    # largest i s.t. p_(i) <= i/n * alpha
    max_i = np.where(below)[0].max()
    cutoff = thresh[max_i]
    mask = p <= cutoff
    return mask


# ---------- main pipeline ----------
def find_multirank_drivers(
    hiddens: np.ndarray,
    tps: Dict[str, int],
    ranks_to_axes: Dict[int, Tuple[int, int]] = {1: (0, 1), 2: (2, 3), 3: (4, 5)},
    start_end_map: Dict[int, Tuple[str, str]] = None,
    n_components: int = 8,
    top_k_candidates: int = 50,
    eps_scale: float = 1e-2,
    n_boot: int = 200,
    alpha: float = 0.05,
    fig_dir: Path = Path("./figs"),
) -> Dict:
    """
    hiddens: shape (n_samples, tot_len, n_hid)
    tps: dict of timepoints (as described by you)
    start_end_map: if None, defaults to stim_r_start : stim_r_off (start) and delay_start:delay_end (end)
    returns dictionary with results and saves figures/CSVs to fig_dir
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_samples, tot_len, n_hid = hiddens.shape
    ranks = sorted(ranks_to_axes.keys())
    if start_end_map is None:
        start_end_map = {r: (f"stim{r}_start", f"stim{r}_off") for r in ranks}
    # end window is delay region
    end_window = (tps["delay_start"], tps["delay_end"] - 2)

    # 1) for each rank compute baseline rotation between start-window and end-window
    baseline_thetas = {}
    Vstart_dict = {}
    Vend_dict = {}
    for r in ranks:
        sname, ename = start_end_map[r]
        s0, s1 = tps[sname], tps[ename]
        e0, e1 = end_window
        window_start = hiddens[:, s0:s1, :]
        window_end = hiddens[:, e0:e1, :]
        V1 = components_from_window(
            window_start, n_components=n_components
        )  # n_hid x n_components
        V2 = components_from_window(window_end, n_components=n_components)
        # pick only the axes that form this rank subspace (e.g. PCs 0,1 -> indices 0,1)
        idx0, idx1 = ranks_to_axes[r]
        Vs1 = V1[:, idx0 : idx1 + 1]
        Vs2 = V2[:, idx0 : idx1 + 1]
        theta = rotation_metric_between_bases(Vs1, Vs2)
        baseline_thetas[r] = float(theta)
        Vstart_dict[r] = Vs1
        Vend_dict[r] = Vs2
        print(f"[rank{r}] baseline theta = {theta:.6f}")

    # 2) finite-difference sensitivity per neuron, per rank
    sensitivities = np.zeros((len(ranks), n_hid))
    for ir, r in enumerate(ranks):
        sname, ename = start_end_map[r]
        s0, s1 = tps[sname], tps[ename]
        e0, e1 = end_window
        window_start = hiddens[:, s0:s1, :]  # shape (n_s, len_s, n_hid)
        window_end = hiddens[:, e0:e1, :]
        # baseline Vstart and Vend recomputed for robustness
        V1 = components_from_window(window_start, n_components=n_components)
        V2 = components_from_window(window_end, n_components=n_components)
        Vs1 = V1[:, ranks_to_axes[r][0] : ranks_to_axes[r][1] + 1]
        Vs2 = V2[:, ranks_to_axes[r][0] : ranks_to_axes[r][1] + 1]
        theta_base = rotation_metric_between_bases(Vs1, Vs2)
        # eps per neuron: small fraction of neuron's std across samples/time in start window
        per_neuron_std = window_start.reshape(-1, n_hid).std(axis=0)
        eps = np.maximum(per_neuron_std * eps_scale, 1e-8)
        # perturb neuron i by +eps across all samples/time in start window
        for i in range(n_hid):
            ws_pert = window_start.copy()
            ws_pert[:, :, i] += eps[i]
            # recompute V1 perturbed
            V1p = components_from_window(ws_pert, n_components=n_components)
            Vs1p = V1p[:, ranks_to_axes[r][0] : ranks_to_axes[r][1] + 1]
            theta_p = rotation_metric_between_bases(Vs1p, Vs2)
            sensitivities[ir, i] = (theta_p - theta_base) / eps[i]
        print(f"[rank{r}] sensitivity computed (shape {sensitivities[ir].shape})")

    # aggregate sensitivities across ranks: use mean absolute sensitivity
    abs_mean_sens = np.mean(np.abs(sensitivities), axis=0)
    # pick top_k candidates
    topk_idx = np.argsort(abs_mean_sens)[::-1][:top_k_candidates]
    print("Top-k candidate neurons (by mean |sensitivity|):", topk_idx.tolist())

    # 3) For topk candidates: LONO + bootstrap per rank
    results = {}
    for ir, r in enumerate(ranks):
        sname, ename = start_end_map[r]
        s0, s1 = tps[sname], tps[ename]
        e0, e1 = end_window
        window_start_all = hiddens[:, s0:s1, :]
        window_end_all = hiddens[:, e0:e1, :]

        # prepare storage
        per_neuron_boot_deltas = np.zeros((len(topk_idx), n_boot))
        per_neuron_delta_mean = np.zeros(len(topk_idx))

        for ni, neuron in enumerate(topk_idx):
            # compute deterministic LONO on full data
            ws_m = window_start_all.copy()
            we_m = window_end_all.copy()
            # replace neuron's activity with its mean across samples/time in that window
            mean_val_start = ws_m[:, :, neuron].mean()
            mean_val_end = we_m[:, :, neuron].mean()
            ws_m[:, :, neuron] = mean_val_start
            we_m[:, :, neuron] = mean_val_end

            # store delta_full as initial point (we will compute bootstrap distribution)
            # Bootstrap: resample trials with replacement, recompute theta and theta_minus
            for b in range(n_boot):
                idxs = np.random.randint(0, n_samples, size=n_samples)
                ws_bs = window_start_all[idxs].copy()
                we_bs = window_end_all[idxs].copy()
                # baseline on bootstrap sample
                V1b = components_from_window(ws_bs, n_components=n_components)
                V2b = components_from_window(we_bs, n_components=n_components)
                Vs1b = V1b[:, ranks_to_axes[r][0] : ranks_to_axes[r][1] + 1]
                Vs2b = V2b[:, ranks_to_axes[r][0] : ranks_to_axes[r][1] + 1]
                theta_b = rotation_metric_between_bases(Vs1b, Vs2b)
                # LONO on bootstrap sample: replace neuron's activity with its mean in the bootstrap sample
                mean_val_bs_start = ws_bs[:, :, neuron].mean()
                mean_val_bs_end = we_bs[:, :, neuron].mean()
                ws_bs[:, :, neuron] = mean_val_bs_start
                we_bs[:, :, neuron] = mean_val_bs_end
                V1bm = components_from_window(ws_bs, n_components=n_components)
                V2bm = components_from_window(we_bs, n_components=n_components)
                Vs1bm = V1bm[:, ranks_to_axes[r][0] : ranks_to_axes[r][1] + 1]
                Vs2bm = V2bm[:, ranks_to_axes[r][0] : ranks_to_axes[r][1] + 1]
                theta_b_minus = rotation_metric_between_bases(Vs1bm, Vs2bm)
                per_neuron_boot_deltas[ni, b] = theta_b - theta_b_minus
            per_neuron_delta_mean[ni] = per_neuron_boot_deltas[ni].mean()
            # progress print
            if (ni + 1) % 10 == 0 or ni == len(topk_idx) - 1:
                print(
                    f"rank{r}: processed LONO+bootstrap for candidate {ni+1}/{len(topk_idx)}"
                )

        # compute p-values: one-sided test H0: delta <= 0
        # p = proportion of bootstrap deltas <= 0
        pvals = np.mean(per_neuron_boot_deltas <= 0.0, axis=1)
        # BH correction across topk for this rank
        reject_mask = benjamini_hochberg(pvals, alpha=alpha)

        # summary table for this rank
        rank_results = []
        for ii, neuron in enumerate(topk_idx):
            mean_delta = float(per_neuron_delta_mean[ii])
            ci_lower = float(np.percentile(per_neuron_boot_deltas[ii], 2.5))
            ci_upper = float(np.percentile(per_neuron_boot_deltas[ii], 97.5))
            pval = float(pvals[ii])
            sig = bool(reject_mask[ii])
            rank_results.append(
                {
                    "neuron": int(neuron),
                    "mean_delta": mean_delta,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "pval": pval,
                    "significant": sig,
                }
            )

        # save CSV
        csv_path = fig_dir / f"rank{r}_lono_boot_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rank_results[0].keys()))
            writer.writeheader()
            for row in rank_results:
                writer.writerow(row)
        print(f"Saved rank {r} LONO bootstrap results to {csv_path}")

        # plot top-k mean delta with CI
        neurons_plot = [row["neuron"] for row in rank_results]
        means_plot = [row["mean_delta"] for row in rank_results]
        lowers = [row["mean_delta"] - row["ci_lower"] for row in rank_results]
        uppers = [row["ci_upper"] - row["mean_delta"] for row in rank_results]

        plt.figure(figsize=(10, 4))
        x = np.arange(len(neurons_plot))
        plt.errorbar(x, means_plot, yerr=[lowers, uppers], fmt="o", capsize=3)
        for xi, sig in enumerate(reject_mask):
            if sig:
                plt.scatter(
                    xi,
                    means_plot[xi],
                    marker="*",
                    s=80,
                    color="red",
                    label="FDR significant" if xi == 0 else "",
                )
        plt.xticks(x, neurons_plot, rotation=90)
        plt.xlabel("Neuron (top candidates)")
        plt.ylabel("Delta theta (theta - theta_minus)")
        plt.title(f"Rank {r} LONO bootstrap mean delta (top {len(topk_idx)})")
        plt.tight_layout()
        figp = fig_dir / f"rank{r}_lono_boot_top{len(topk_idx)}.png"
        plt.savefig(figp, dpi=150)
        plt.close()
        print(f"Saved figure {figp}")

        # store results
        results[r] = {
            "baseline_theta": baseline_thetas[r],
            "topk_idx": topk_idx,
            "rank_results": rank_results,
            "pvals": pvals,
            "reject_mask": reject_mask,
            "boot_deltas": per_neuron_boot_deltas,
        }

    # 4) find neurons significant across multiple ranks
    # n_ranks = len(ranks)
    sig_counts = np.zeros(n_hid, dtype=int)
    for r in ranks:
        mask = np.zeros(n_hid, dtype=bool)
        for entry in results[r]["rank_results"]:
            idx = entry["neuron"]
            if entry["significant"]:
                mask[idx] = True
        sig_counts += mask.astype(int)

    multi_rank_neurons = np.where(sig_counts >= 2)[0]  # appear in >=2 ranks
    # save multi-rank table
    multi_csv = fig_dir / "multi_rank_significant_neurons.csv"
    with open(multi_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["neuron", "n_ranks_significant"])
        for n in multi_rank_neurons:
            writer.writerow([int(n), int(sig_counts[n])])
    print("Neurons significant in >=2 ranks:", multi_rank_neurons.tolist())
    print(f"Saved multi-rank summary to {multi_csv}")

    results["multi_rank_neurons"] = multi_rank_neurons
    results["sig_counts"] = sig_counts
    results["topk_idx"] = topk_idx
    return results


# === Usage example ===
if __name__ == "__main__":
    TASK = "forward"
    N_SAMPLES = 120
    FILENAME = Path(__file__).name
    params = utils.initialize_analysis_legacy(TASK, N_SAMPLES, FILENAME)
    outputs_dict = params["OUTPUTS_DICT"]
    tps = params["TIMEPOINTS"]
    hiddens = outputs_dict["hiddens"]
    hiddens = np.tanh(hiddens)  # keep your original transform

    FIG_DIR = Path(params["FIG_DIR"]) / "rank_rotation_analysis"
    # res = find_multirank_drivers(
    #     hiddens=hiddens,
    #     tps=tps,
    #     ranks_to_axes={1:(0,1), 2:(2,3), 3:(4,5)},
    #     n_components=8,
    #     top_k_candidates=50,
    #     eps_scale=1e-2,
    #     n_boot=200,
    #     alpha=0.05,
    #     fig_dir=FIG_DIR
    # )

    res = find_multirank_drivers(
        hiddens=hiddens,
        tps=tps,
        ranks_to_axes={1: (0, 1), 2: (2, 3), 3: (4, 5)},
        n_components=8,
        top_k_candidates=50,
        eps_scale=1e-2,
        n_boot=200,
        alpha=0.05,
        fig_dir=FIG_DIR,
    )

    # print short summary
    print("=== Summary ===")
    for r in sorted([k for k in res.keys() if isinstance(k, int)]):
        sigs = [row for row in res[r]["rank_results"] if row["significant"]]
        print(
            f"Rank {r}: baseline theta = {res[r]['baseline_theta']:.5f}, significant neurons (count) = {len(sigs)}"
        )
    print("Neurons significant in >=2 ranks:", res["multi_rank_neurons"].tolist())

    r"""
    Results:
    Extension for Scikit-learn* enabled (https://github.com/uxlfoundation/scikit-learn-intelex)
    [INFO] Load model from: D:\CodeWork\CodeTest\schema\model\temp\forward_latest.pth
    [rank1] baseline theta = 2.149105
    [rank2] baseline theta = 2.052465
    [rank3] baseline theta = 2.134454
    [rank1] sensitivity computed (shape (400,))
    [rank2] sensitivity computed (shape (400,))
    [rank3] sensitivity computed (shape (400,))
    Top-k candidate neurons (by mean |sensitivity|): [56, 195, 193, 317, 395, 147, 206, 297, 253, 373, 48, 330, 348, 277, 42, 144, 294, 303, 118, 160, 146, 308, 39, 97, 170, 186, 74, 382, 63, 287, 346, 267, 367, 77, 203, 334, 390, 168, 152, 32, 154, 386, 105, 347, 194, 218, 349, 366, 282, 213]
    rank1: processed LONO+bootstrap for candidate 10/50
    rank1: processed LONO+bootstrap for candidate 20/50
    rank1: processed LONO+bootstrap for candidate 30/50
    rank1: processed LONO+bootstrap for candidate 40/50
    rank1: processed LONO+bootstrap for candidate 50/50
    Saved rank 1 LONO bootstrap results to D:\CodeWork\CodeTest\schema\figure\overlapping_neuron_analysis\rank_rotation_analysis\rank1_lono_boot_results.csv
    Saved figure D:\CodeWork\CodeTest\schema\figure\overlapping_neuron_analysis\rank_rotation_analysis\rank1_lono_boot_top50.png
    rank2: processed LONO+bootstrap for candidate 10/50
    rank2: processed LONO+bootstrap for candidate 20/50
    rank2: processed LONO+bootstrap for candidate 30/50
    rank2: processed LONO+bootstrap for candidate 40/50
    rank2: processed LONO+bootstrap for candidate 50/50
    Saved rank 2 LONO bootstrap results to D:\CodeWork\CodeTest\schema\figure\overlapping_neuron_analysis\rank_rotation_analysis\rank2_lono_boot_results.csv
    Saved figure D:\CodeWork\CodeTest\schema\figure\overlapping_neuron_analysis\rank_rotation_analysis\rank2_lono_boot_top50.png
    rank3: processed LONO+bootstrap for candidate 10/50
    rank3: processed LONO+bootstrap for candidate 20/50
    rank3: processed LONO+bootstrap for candidate 30/50
    rank3: processed LONO+bootstrap for candidate 40/50
    rank3: processed LONO+bootstrap for candidate 50/50
    Saved rank 3 LONO bootstrap results to D:\CodeWork\CodeTest\schema\figure\overlapping_neuron_analysis\rank_rotation_analysis\rank3_lono_boot_results.csv
    Saved figure D:\CodeWork\CodeTest\schema\figure\overlapping_neuron_analysis\rank_rotation_analysis\rank3_lono_boot_top50.png
    Neurons significant in >=2 ranks: []
    Saved multi-rank summary to D:\CodeWork\CodeTest\schema\figure\overlapping_neuron_analysis\rank_rotation_analysis\multi_rank_significant_neurons.csv
    === Summary ===
    Rank 1: baseline theta = 2.14911, significant neurons (count) = 1
    Rank 2: baseline theta = 2.05247, significant neurons (count) = 1
    Rank 3: baseline theta = 2.13445, significant neurons (count) = 0
    Neurons significant in >=2 ranks: []
    """
