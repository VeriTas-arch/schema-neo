"""
Core utility functions for training and evaluation. For simplicity, all functions are
defined in this single file. If one wants a more extensible codebase, it is recommended to
split functions into multiple files according to their functionalities.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rich.logging import RichHandler

from data.ab_synthetic_gen import ITEM_TEMPLATES, N_STIM

# Set matplotlib style once at module import (avoid repeated calls)
plt.style.use("default")

# Precompute helper values from ITEM_TEMPLATES to avoid repeated work
# PEAK_CHANNELS: the channel index within the 24d stimulus that peaks for
# each of the 6 item templates. TEMPLATE_NORMS: l2 norm of each template.
PEAK_CHANNELS = np.argmax(ITEM_TEMPLATES, axis=1)
TEMPLATE_NORMS = np.linalg.norm(ITEM_TEMPLATES, axis=1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#! ------------- MISC HELP FUNCTIONS -------------
def set_seed(seed, cuda=True):
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Seed value used for all RNGs.
    cuda : bool, optional
        Whether to also set CUDA's random seed (if available), by default True.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def get_logger(log_level):
    """
    Create a logger configured with RichHandler.

    Parameters
    ----------
    log_level : int
        Logging level (e.g., ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    rich_handler = RichHandler(rich_tracebacks=True, show_time=True)
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)

    logger.addHandler(rich_handler)

    return logger


def zero_mask(mask, i, start, length):
    """
    Zero-out a contiguous segment of a mask for a given sample.

    Parameters
    ----------
    mask : torch.Tensor
        Mask tensor of shape ``(B, T)``.
    i : int
        Sample index (batch dimension).
    start : int
        Start time index (inclusive).
    length : int
        Length of the segment to be zeroed.
    """
    mask[i, start : start + length] = 0


#! ------------- TRAINING FUNCTIONS -------------
def validate(model, val_loader, cfg):
    """
    Evaluate the model on a validation loader.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    val_loader : torch.utils.data.DataLoader
        DataLoader yielding validation batches.
    cfg : Any
        Configuration object used by ``batch_processor()``.

    Returns
    -------
    float
        Mean validation loss over all batches.
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for _, data in enumerate(val_loader):
            output = batch_processor(model, data, cfg)

            total_loss += output["record_res"]["loss"]
            num_batches += 1

    return total_loss / num_batches


def batch_processor(model, data, cfg):
    """
    Process a single batch: forward pass, masked loss, and logging payload.

    Parameters
    ----------
    model : torch.nn.Module
        RNN model receiving inputs of shape ``(T, B, D)``.
    data : tuple
        Batch tuple ``(data_x, data_y, data_t, data_reg)`` where
        ``data_x`` has shape ``(B, T, D)``, ``data_y`` has shape ``(B,)``,
        ``data_t`` has shape ``(B, 8)`` indicating timepoints for each sample, and
        ``data_reg`` is the target of shape ``(B, T, K)``.
    cfg : Any
        Configuration with attributes used inside the function (e.g., ``h0_trainable``).

    Returns
    -------
    dict
        Dictionary including ``loss`` (tensor) and ``record_res`` (dict with numpy arrays).
    """
    data_x, data_y, data_t, data_reg, data_seq_len = data
    batch = data_x.shape[0]

    data_x = data_x.transpose(1, 0).to(DEVICE)
    data_y = data_y.to(DEVICE)
    data_reg = data_reg.to(DEVICE)

    if cfg.h0_trainable:
        h0 = model.h0.data.repeat(batch, 1).to(DEVICE)
    else:
        h0 = None

    final_out, hiddens = model(data_x, h0)
    final_out = final_out.transpose(1, 0)  # (batch, T, output_size)

    # unpack hiddens
    hidden_activity = hiddens[1]  # (batch, T, hidden_size)

    data_t = np.squeeze(data_t, axis=1)
    batch_size, T = final_out.shape[:2]

    mask = build_mask(data_t, data_seq_len, batch_size, T)  # (batch, T)
    mask = mask.unsqueeze(-1).expand_as(final_out).clone()
    mask[:, :, 0:6] *= 5  # larger weight for non-fixation outputs
    mask = mask.to(DEVICE)

    final_out_masked = final_out * mask
    data_reg_masked = data_reg * mask

    task_loss = F.mse_loss(final_out_masked, data_reg_masked, reduction="none")
    denom = mask.sum()
    denom = denom.clamp_min(1e-5)
    task_loss = (task_loss * mask).sum() / denom

    l2_lambda_weights = 0.0008
    l2_reg = sum(torch.norm(param, 2) for param in model.parameters())

    # total loss
    loss = task_loss + l2_lambda_weights * l2_reg

    # fmt: off
    record_res = OrderedDict()
    record_res["timepoints"]    =   data_t.cpu().detach().numpy()
    record_res["seq_lens"]      =   data_seq_len.cpu().detach().numpy()
    record_res["outputs"]       =   final_out.cpu().detach().numpy()
    record_res["labels"]        =   data_y.cpu().detach().numpy()
    record_res["inputs"]        =   data_x.transpose(1, 0).cpu().detach().numpy()
    record_res["targets"]       =   data_reg.cpu().detach().numpy()
    record_res["hiddens"]       =   hidden_activity.cpu().detach().numpy()
    record_res["loss"]          =   loss.item()
    # fmt: on

    outputs = dict(loss=loss, record_res=record_res)
    return outputs


def load_pretrained_model(model, pretrained_path, freeze_except=None):
    """
    Load a pretrained model and optionally freeze layers.

    Parameters
    ----------
    model : torch.nn.Module
        Model into which weights will be loaded.
    pretrained_path : str | pathlib.Path
        Path to the pretrained model checkpoint.
    freeze_except : list[str] | None, optional
        Keep only these layers trainable (others are frozen). If ``None`` (default),
        all parameters remain trainable.

    Returns
    -------
    torch.nn.Module
        Model with loaded weights and updated ``requires_grad`` flags.
    """
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        if (freeze_except is None) or (name.split(".")[-1] in freeze_except):
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(f"{name}: requires_grad={param.requires_grad}")

    return model


def build_mask(data_t, data_seq_len, batch_size, T):
    """
    Build a per-time-step loss mask based on trial timepoints.

    Parameters
    ----------
    data_t : np.ndarray
        Array of shape ``(B, 8)`` with timepoints ``(t1..t8)`` for each sample.
    data_seq_len : np.ndarray
        Array of shape ``(B,)`` with sequence lengths for each sample.
    batch_size : int
        Batch size.
    T : int
        Total number of time steps.

    Returns
    -------
    torch.Tensor
        Mask tensor of shape ``(B, T)`` where zeros indicate steps excluded from loss.
    """
    mask = torch.ones(batch_size, T, dtype=torch.float32)

    for i in range(batch_size):
        # t1: stimulus 1 start, t2: stimulus 2 start, t3: stimulus 3 start
        # t4: pre_cue start, t5: pre cue length, t6: post cue length
        # t7: retrieval start, t8: target length
        t1, t2, t3, t4, t5, t6, t7, t8 = data_t[i]
        # seq_len = int(data_seq_len[i])
        #! seq_len dependent masking seems not work well here
        #! perhaps the random target length with everything beyond
        #! masked confuses the model
        max_seq_len = 3

        # stimulus period masking
        zero_mask(mask, i, t1, 2)
        zero_mask(mask, i, t2, 2)
        zero_mask(mask, i, t3, 2)

        # target period masking
        for j in range(max_seq_len):
            target_st = t7 + j * t8
            zero_mask(mask, i, target_st, 2)

        # mask everything beyond retrieval period, i.e. we do not compute the loss
        # after the last target
        retrieval_end = t7 + max_seq_len * t8
        mask[i, retrieval_end:] = 0

    return mask


#! ------------- VISUALIZATION HELP FUNCTIONS -------------
def decode_presented_items(
    x: np.ndarray, t: np.ndarray, stim_dur: int = 8, seq_len: int = 3
):
    """
    Decode presented items during stimulus windows via cosine similarity.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape ``(T, N_IN)``; only the first ``N_STIM`` channels are used.
    t : numpy.ndarray
        Timepoint array with stimulus start indices. Should have at least ``seq_len`` entries.
    stim_dur : int, optional
        Duration of each stimulus window, by default 8.
    seq_len : int, optional
        Number of stimuli in the sequence, by default 3.

    Returns
    -------
    presented_items : list[int]
        Detected item indices of length ``seq_len`` (``-1`` if not detected).
    """
    x_arr = np.asarray(x)
    t_vals = np.asarray(t)

    # t1: stimulus 1 start, t2: stimulus 2 start, t3: stimulus 3 start
    # t4: pre_cue start, t5: pre cue length, t6: post cue length
    # t7: retrieval start, t8: target length
    t1, t2, t3, t4, t5, t6, t7, t8 = [int(v) for v in t_vals]

    # detect presented items
    presented_items = []
    for stim_start in [t1, t2, t3][:seq_len]:
        win = x_arr[stim_start : stim_start + stim_dur, :N_STIM]
        mean_vec = np.mean(win, axis=0)
        mean_norm = np.linalg.norm(mean_vec)
        tmpl_norms = TEMPLATE_NORMS

        denom = tmpl_norms * mean_norm
        valid_denom = denom > 0
        scores = np.divide(ITEM_TEMPLATES @ mean_vec, denom, where=valid_denom)

        presented_items.append(int(np.argmax(scores)))

    return presented_items


def decode_cue_type(cue_fixation_inputs: np.ndarray, task: str = None):
    """
    Decode cue direction (forward/backward) from cue channels.

    Parameters
    ----------
    cue_fixation_inputs : numpy.ndarray
        fixation and cue (if exists) channels of shape ``(T, n_order)`` (e.g., cue1, cue2, fixation).
    task : str, optional
        Task type, by default None.

    Returns
    -------
    str
        One of {"forward", "backward", "unknown", "N/A"}.
    """
    # require switch task and valid timepoints
    if task != "switch":
        return "N/A"

    cue1_channel = cue_fixation_inputs[:, 0]
    cue2_channel = cue_fixation_inputs[:, 1]

    #! since cue channel has noise added, we use a threshold to determine
    if np.all(cue1_channel < 0.5):
        return "forward"
    elif np.all(cue2_channel < 0.5):
        return "backward"
    else:
        return "unknown"


def plot_model_performance(
    record_res, sample_index, stim_dur: int = 8, task: str = "N/A", fig_dir=None
):
    """
    Plot inputs, targets, and model outputs for a single sample.

    Parameters
    ----------
    record_res : dict
        Dictionary containing numpy arrays for keys: ``inputs``, ``outputs``, ``targets``,
        and ``timepoints``. See ``batch_processor()`` for details.
    sample_index : int
        Index of the sample within the current batch to visualize.
    stim_dur : int, optional
        Stimulus presentation duration, by default 8.
    task : str, optional
        Task type ("switch", "forward", "backward"), by default "N/A".
    fig_dir : str | pathlib.Path | None, optional
        If provided, the figure is saved to this path; otherwise only returned.

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure, or ``None`` if ``sample_index`` is out of range.
    """
    # bound check: some hooks may ask more samples than batch size
    if "inputs" not in record_res or sample_index >= len(record_res["inputs"]):
        # nothing to plot
        return None

    # local reference to avoid repeated global lookups
    _PEAK_CHANNELS = PEAK_CHANNELS
    _N_STIM = N_STIM

    inputs = record_res["inputs"][sample_index]
    inputs_item = inputs[:, :_N_STIM]  # First 24 channels
    cue_fixation_inputs = inputs[:, _N_STIM:]  # Remaining channels (cue + fixation)
    outputs = record_res["outputs"][sample_index]
    targets = record_res["targets"][sample_index]
    t_arr = record_res["timepoints"][sample_index]
    seq_len = int(record_res["seq_lens"][sample_index])

    o_dim = outputs.shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax_in, ax_order, ax_target, ax_out = axes

    # decode presented items
    presented = decode_presented_items(
        inputs_item, t_arr, stim_dur=stim_dur, seq_len=seq_len
    )

    # color scheme
    ITEM_COLORS = [
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]
    FIXATION_COLOR = "tab:blue"

    # sub figure 1: input items (show detected items' peak channels)
    plotted = set()
    T = inputs_item.shape[0]
    for idx, item in enumerate(presented):
        if item is None or item < 0:
            continue
        peak_ch = int(_PEAK_CHANNELS[item])
        if peak_ch in plotted:
            continue
        plotted.add(peak_ch)
        ax_in.plot(
            np.arange(T),
            inputs_item[:, peak_ch],
            color=ITEM_COLORS[item],
            label=f"item{item} (ch{peak_ch})",
        )
    ax_in.set_title(
        "Input Items (decoded presented items)" if plotted else "Input Items"
    )
    ax_in.set_ylabel("Activation")
    if plotted:
        ax_in.legend(loc="upper right")

    # sub figure 2: fixation/cue channels
    cue_type = decode_cue_type(cue_fixation_inputs, task=task)
    ax_order.set_title(
        f"Fixation and Cues (Cue: {cue_type})" if task == "switch" else "Fixation"
    )
    ax_order.set_ylabel("Activation")

    n_order = cue_fixation_inputs.shape[1]  # [cue1, cue2, fixation]
    labels = []
    for j in range(n_order):
        gidx = _N_STIM + j
        if gidx == inputs.shape[1] - 1:
            labels.append("fixation")
            color = FIXATION_COLOR
        else:
            labels.append(f"cue {j}" if task == "switch" else f"ch{gidx}")
            color = ITEM_COLORS[j]
        ax_order.plot(cue_fixation_inputs[:, j], color=color)
    ax_order.legend(labels, loc="upper right", ncol=2)

    # sub figure 3: target outputs
    ax_target.set_title("Target")
    ax_target.set_ylabel("Activation")
    for j in range(o_dim):
        s = targets[:, j]
        if np.allclose(s, 0.0, atol=1e-12):
            continue
        color = FIXATION_COLOR if j == o_dim - 1 else ITEM_COLORS[j]
        label = "fixation" if j == o_dim - 1 else f"item {j}"
        ax_target.plot(s, color=color, label=label)
    ax_target.legend(ncol=2, loc="upper right")

    # sub figure 4: model outputs
    ax_out.set_title("Model Output")
    ax_out.set_ylabel("Activation")
    ax_out.set_xlabel("Time (steps)")
    for j in range(o_dim):
        s = outputs[:, j]
        if np.allclose(s, 0.0, atol=1e-12):
            continue
        color = FIXATION_COLOR if j == o_dim - 1 else ITEM_COLORS[j]
        label = "fixation" if j == o_dim - 1 else f"item {j}"
        ax_out.plot(s, color=color, label=label)
    ax_out.legend(ncol=2, loc="upper right")

    plt.tight_layout()

    if fig_dir is not None:
        fig.savefig(fig_dir, dpi=150, bbox_inches="tight")

    plt.close()

    return fig
