"""
Utility functions for RNN analysis and plotting.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import colorsys
import inspect
import itertools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from jax import jit
from sklearn.decomposition import PCA
from sklearnex import patch_sklearn

from config.ab_synthetic_config import SynConfig
from data.ab_synthetic_gen import abcDataset
from lib.model import RNNConfig, RNNNet, SimpleRNN, LegacyRNNConfig, LegacyRNNNet

patch_sklearn()

#! ------------- DEVICE AND PATH CONSTANTS -------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_BASE = Path(__file__).resolve().parent.parent.parent / "figure"
MODEL_BASE = Path(__file__).resolve().parent.parent.parent / "model"

#! ------------- DEFAULT CONSTANTS -------------
DEFAULT_FRAMEWORK = "jax"
DEFAULT_REGRESS_MODE = "hierarchy_long"
DEFAULT_N_CLASS = 120


#! ------------- LEGACY FUNCTIONS -------------
def inputs_test_jax_legacy(val_loader):
    for j, (inputs, labels, stimes, dseq) in enumerate(val_loader):
        inputs = jnp.array(inputs.cpu().numpy())
        labels = jnp.array(labels.cpu().numpy())
        dseq = jnp.array(dseq.cpu().numpy())
        stimes = jnp.array(stimes.cpu().numpy())
        break

    return inputs, labels, stimes, dseq


def set_params_from_torch_legacy(model_path):
    model = torch.load(model_path, map_location=DEVICE)
    state_dict = model["state_dict"]
    return {
        "Win": state_dict["Win"].cpu().numpy(),
        "Wout": state_dict["Wout"].cpu().numpy(),
        "Wr": state_dict["Wr"].cpu().numpy(),
        "bias": state_dict["bias"].cpu().numpy(),
    }


def get_rnn_outputs_jax_legacy(config, load_from, val_loader, silence_whh=False):
    ## build model
    trained_params = set_params_from_torch_legacy(load_from)
    Win = trained_params["Win"]
    Wout = trained_params["Wout"]

    rnn = LegacyRNNNet(config)

    data_x, data_y, data_t, data_reg = inputs_test_jax_legacy(val_loader)
    batch = data_x.shape[0]
    inputs = data_x.transpose(1, 0, 2)  ## (T, batch, M)
    rnn_init = jnp.zeros((batch, 400))

    if silence_whh:
        silence_Whh(trained_params["Wr"])

    params = dict(params=dict(ScanLegacySimpleRNN_0=trained_params))
    rnn_jit = jit(rnn.apply)

    _, rnn_output = rnn_jit(params, carray=rnn_init, inputs=inputs)

    inputs = inputs.transpose(1, 0, 2)  ## (batch, T, M)
    outputs = rnn_output[1].transpose(1, 0, 2)  ## (batch, T, K) = (120, 48, 3)
    hiddens = rnn_output[0].transpose(1, 0, 2)  ## (batch, T, N_HID) = (120, T, 50)

    outputs_dict: dict = {
        "inputs": data_x,
        "hiddens": hiddens,
        "outputs": outputs,
        "labels": data_y,
        "targets": data_reg,
        "batch": batch,
        "Win": Win,
        "Wout": Wout,
        "trained_params": trained_params,
    }
    return outputs_dict


def initialize_analysis_legacy(task, n_samples=120, filename=None, **kwargs):

    if task == "forward":
        from data.ab_synthetic_gen_forward import forwardDataset as abcDataset
    elif task == "backward":
        from data.ab_synthetic_gen_backward import backwardDataset as abcDataset
    elif task == "switch":
        from data.ab_synthetic_gen_switch import switchDataset as abcDataset
    else:
        raise ValueError("No such task!")

    # suppose the current filename is "subspace_{xxx}.py"
    # extract "xxx" from the filename
    if filename.startswith("subspace_") and filename.endswith(".py"):
        extracted_name = filename[len("subspace_") : -len(".py")]
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

    fig_dir = FIG_BASE / extracted_name
    Path.mkdir(fig_dir, parents=True, exist_ok=True)

    # load kwargs
    silence_index = kwargs.get("silence_index", None)
    silence_whh = kwargs.get("silence_whh", False)

    class_index = np.arange(0, n_samples)

    val_dataset = abcDataset(
        noise_sigma=0,
        stim_dur=4,
        num_data=n_samples,
        stim_123=None,  # stim_l
        class_index=class_index,
        regress_target_mode="hierarchy_long",
        silence_index=silence_index,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=n_samples, shuffle=False, num_workers=0
    )

    config = LegacyRNNConfig
    config.N_hid = 400
    config.N_in = 27 if task == "switch" else 25
    config.N_out = 7
    config.tau = 1.0
    config.dt = 0.05

    load_model_dir = MODEL_BASE / "temp" / f"legacy_{task}_latest.pth"
    outputs_dict = get_rnn_outputs_jax_legacy(
        config, load_model_dir, val_loader, silence_whh=silence_whh
    )

    timepoints = val_dataset.get_timepoints()

    params = {
        "TASK": task,
        "LOAD_MODEL_DIR": load_model_dir,
        "FIG_DIR": fig_dir,
        "N_IN": config.N_in,
        "N_HID": config.N_hid,
        "N_OUT": config.N_out,
        "TOT_LEN": 200,
        "STIM_DUR": 8,
        "STIM_INTERVAL": 8,
        "TARGET_LEN": 12,
        "N_CLASS": DEFAULT_N_CLASS,
        "N_SAMPLES": n_samples,
        "PERMS": list(itertools.permutations(range(6), 3)),
        "CLASS_INDEX": class_index,
        "VAL_LOADER": val_loader,
        "OUTPUTS_DICT": outputs_dict,
        "TIMEPOINTS": timepoints,
        "RNN_CONFIG": config,
    }

    return params


def initialize_analysis_legacy_multi_models(
    model_index, task, n_samples=120, filename=None, **kwargs
):

    if task == "forward":
        from data.ab_synthetic_gen_forward import forwardDataset as abcDataset
    elif task == "backward":
        from data.ab_synthetic_gen_backward import backwardDataset as abcDataset
    elif task == "switch":
        from data.ab_synthetic_gen_switch import switchDataset as abcDataset
    else:
        raise ValueError("No such task!")

    # suppose the current filename is "subspace_{xxx}.py"
    # extract "xxx" from the filename
    if filename.startswith("subspace_") and filename.endswith(".py"):
        extracted_name = filename[len("subspace_") : -len(".py")]
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

    fig_dir = FIG_BASE / extracted_name
    Path.mkdir(fig_dir, parents=True, exist_ok=True)

    # load kwargs
    silence_index = kwargs.get("silence_index", None)
    silence_whh = kwargs.get("silence_whh", False)

    class_index = np.arange(0, n_samples)

    val_dataset = abcDataset(
        noise_sigma=0,
        stim_dur=4,
        num_data=n_samples,
        stim_123=None,  # stim_l
        class_index=class_index,
        regress_target_mode="hierarchy_long",
        silence_index=silence_index,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=n_samples, shuffle=False, num_workers=0
    )

    config = LegacyRNNConfig
    config.N_hid = 400
    config.N_in = 27 if task == "switch" else 25
    config.N_out = 7
    config.tau = 1.0
    config.dt = 0.05

    all_model_suffix = ["1106", "1107", "1108", "1109", "1110"]
    model_suffix = all_model_suffix[model_index]
    load_model_dir = MODEL_BASE / "temp" / f"legacy_{task}_{model_suffix}.pth"

    outputs_dict = get_rnn_outputs_jax_legacy(
        config, load_model_dir, val_loader, silence_whh=silence_whh
    )

    timepoints = val_dataset.get_timepoints()

    params = {
        "TASK": task,
        "LOAD_MODEL_DIR": load_model_dir,
        "FIG_DIR": fig_dir,
        "N_IN": config.N_in,
        "N_HID": config.N_hid,
        "N_OUT": config.N_out,
        "TOT_LEN": 200,
        "STIM_DUR": 8,
        "STIM_INTERVAL": 8,
        "TARGET_LEN": 12,
        "N_CLASS": DEFAULT_N_CLASS,
        "N_SAMPLES": n_samples,
        "PERMS": list(itertools.permutations(range(6), 3)),
        "CLASS_INDEX": class_index,
        "VAL_LOADER": val_loader,
        "OUTPUTS_DICT": outputs_dict,
        "TIMEPOINTS": timepoints,
        "RNN_CONFIG": config,
    }

    return params


#! ------------- MODEL PRE-PROCESS FUNCTIONS -------------
def get_config_from_model(model_path):
    """
    return the training config dataclass
    """
    model = torch.load(model_path, map_location=DEVICE)
    config = model["config"]
    config_dataclass = SynConfig(**config)
    return config_dataclass


def set_params_from_torch(model_path):
    state_dict = torch.load(model_path, map_location=DEVICE)["model_state_dict"]
    return {
        "Win": state_dict["Win"].cpu().numpy(),
        "Wout": state_dict["Wout"].cpu().numpy(),
        "Wr": state_dict["Wr"].cpu().numpy(),
        "bias": state_dict["bias"].cpu().numpy(),
    }


def get_rnn_outputs(config, load_from, val_loader, framework="jax"):
    if framework == "jax":
        return get_rnn_outputs_jax(config, load_from, val_loader)
    elif framework == "torch":
        return get_rnn_outputs_torch(config, load_from, val_loader)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def inputs_test_torch(val_loader):
    for _, (data_x, data_y, data_t, data_reg, data_seq_len) in enumerate(val_loader):
        data_x = data_x.detach().cpu().numpy()
        data_y = data_y.detach().cpu().numpy()
        data_t = data_t.detach().cpu().numpy()
        data_reg = data_reg.detach().cpu().numpy()
        data_seq_len = data_seq_len.detach().cpu().numpy()
        break
    return data_x, data_y, data_t, data_reg, data_seq_len


def inputs_test_jax(val_loader):
    for _, (data_x, data_y, data_t, data_reg, data_seq_len) in enumerate(val_loader):
        data_x = jnp.array(data_x.detach().cpu().numpy())
        data_y = jnp.array(data_y.detach().cpu().numpy())
        data_t = jnp.array(data_t.detach().cpu().numpy())
        data_reg = jnp.array(data_reg.detach().cpu().numpy())
        data_seq_len = jnp.array(data_seq_len.cpu().numpy())
        break
    return data_x, data_y, data_t, data_reg, data_seq_len


def silence_Whh(Wr):
    # fmt: off
    # neuron_list = [105, 175, 269, 13, 368, 343, 146, 302, 364, 31, 22, 334, 67, 389, 14, 99, 7, 207, 111, 253]
    neuron_list = [105, 175, 269, 13, 368, 343, 146, 302, 364, 31, 22, 334, 67, 389, 14, 99, 7, 207, 111, 253, 60, 140, 144, 117, 388, 1, 16, 147, 58, 358, 162, 195, 252, 396, 19, 315, 71, 366, 277, 311, 4, 296, 84, 201, 169, 230, 383, 300, 64, 318]
    # neuron_list = np.array([i for i in range(400)])  # silence all neurons
    # fmt: on

    for idx in neuron_list:
        Wr[:, idx] = 0.0
        Wr[idx, :] = 0.0


def get_rnn_outputs_torch(config, load_from, val_loader):
    rnn = SimpleRNN(**config)
    rnn = rnn.to(DEVICE)
    state_dict = torch.load(load_from, map_location=DEVICE)["model_state_dict"]

    Win = state_dict["Win"].cpu().numpy()
    Wout = state_dict["Wout"].cpu().numpy()

    rnn.load_state_dict(state_dict, strict=False)
    rnn.eval()

    data_x, data_y, data_t, data_reg, data_seq_len = inputs_test_torch(val_loader)
    batch = data_x.shape[0]  # inputs shape: (batch, T, N_IN)
    inputs_torch = torch.tensor(data_x, dtype=torch.float32, device=DEVICE).transpose(
        0, 1
    )  # (T, batch, N_IN)

    hidden_init = rnn.init_hidden(batch)

    with torch.no_grad():
        output_tensor, (tot_input_tensor, tot_rnnhid_tensor, tot_output_tensor) = rnn(
            inputs_torch, hidden_init=hidden_init
        )

    hiddens = tot_rnnhid_tensor.detach().cpu().numpy()  # (batch, T, N_HID)
    outputs = tot_output_tensor.detach().cpu().numpy()  # (batch, T, N_OUT)

    outputs_dict: dict = {
        "inputs": data_x,
        "hiddens": hiddens,
        "outputs": outputs,
        "labels": data_y,
        "targets": data_reg,
        "batch": batch,
        "Win": Win,
        "Wout": Wout,
        "trained_params": state_dict,
    }

    return outputs_dict


def get_rnn_outputs_jax(config: dict, load_from, val_loader):
    trained_params = set_params_from_torch(load_from)
    Wout = trained_params["Wout"]
    Win = trained_params["Win"]

    rnnconfig_keys = inspect.signature(RNNConfig).parameters.keys()
    config_filtered = {k: v for k, v in config.items() if k in rnnconfig_keys}

    config = RNNConfig(**config_filtered)
    rnn = RNNNet(config)

    data_x, data_y, data_t, data_reg, data_seq_len = inputs_test_jax(val_loader)
    batch = data_x.shape[0]
    data_x = data_x.transpose(1, 0, 2)  ## (T, batch, M)
    rnn_init = jnp.zeros((batch, config.hidden_size))

    params = dict(params=dict(ScanSimpleJaxRNN_0=trained_params))
    rnn_jit = jit(rnn.apply)

    _, rnn_output = rnn_jit(params, carray=rnn_init, inputs=data_x)

    data_x = data_x.transpose(1, 0, 2)  ## (batch, T, M)
    outputs = rnn_output[1].transpose(1, 0, 2)  ## (batch, T, K) = (120, 48, 3)
    hiddens = rnn_output[0].transpose(1, 0, 2)  ## (batch, T, N_HID) = (120, T, 50)

    outputs_dict: dict = {
        "inputs": data_x,
        "hiddens": hiddens,
        "outputs": outputs,
        "labels": data_y,
        "targets": data_reg,
        "batch": batch,
        "Win": Win,
        "Wout": Wout,
        "trained_params": trained_params,
    }
    return outputs_dict


#! ------------- PLOT FUNCTIONS -------------
def set_plot(ll=7):
    """
    Set plotting parameters. Returns colors for plots

    Parameters
    ----------
    ll : int, optional
        Number of colors. 5 or 7. The default is 7.

    Returns
    -------
    clS : colors

    """
    plt.style.use("ggplot")

    fig_width = 1.5 * 2.2 * 1.5  # width in inches
    fig_height = 1.5 * 2 * 1.5  # height in inches
    fig_size = [fig_width, fig_height]
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.autolayout"] = True

    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["lines.markeredgewidth"] = 0.003
    plt.rcParams["lines.markersize"] = 3
    plt.rcParams["font.size"] = 14  # 9
    plt.rcParams["legend.fontsize"] = 11  # 7.
    plt.rcParams["axes.facecolor"] = "1"
    plt.rcParams["axes.edgecolor"] = "0"
    plt.rcParams["axes.linewidth"] = "0.7"

    plt.rcParams["axes.labelcolor"] = "0"
    plt.rcParams["axes.labelsize"] = 14  # 9
    plt.rcParams["xtick.labelsize"] = 11  # 7
    plt.rcParams["ytick.labelsize"] = 11  # 7
    plt.rcParams["xtick.color"] = "0"
    plt.rcParams["ytick.color"] = "0"
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2

    plt.rcParams["font.sans-serif"] = "Arial"

    clS = np.zeros((ll, 3))

    cl11 = np.array((102, 153, 255)) / 255.0
    cl12 = np.array((53, 153, 53)) / 255.0

    cl21 = np.array((255, 204, 51)) / 255.0
    cl22 = np.array((204, 0, 0)) / 255.0

    if ll == 7:
        clS[0, :] = 0.4 * np.ones((3,))

        clS[1, :] = cl11
        clS[2, :] = 0.5 * cl11 + 0.5 * cl12
        clS[3, :] = cl12

        clS[4, :] = cl21
        clS[5, :] = 0.5 * cl21 + 0.5 * cl22
        clS[6, :] = cl22

        clS = clS[1:]
        clS = clS[::-1]

        c2 = [67 / 256, 90 / 256, 162 / 256]
        c1 = [220 / 256, 70 / 256, 51 / 256]
        clS[0, :] = c1
        clS[5, :] = c2
    elif ll == 5:
        clS[0, :] = 0.4 * np.ones((3,))

        clS[2, :] = cl12

        clS[3, :] = cl21

        clS[4, :] = cl22
    return clS


#! ------------- ANALYSIS HELP FUNCTIONS -------------
def initialize_analysis(task, n_samples=120, filename=None, **kwargs):
    r"""
    Initialize analysis parameters and return a dictionary of relevant variables.

    Parameters
    ----------
    task : str
        Task type, e.g., "switch", "forward", "backward".
    n_samples : int, optional
        Number of trajectories / data to analyze. The default is 120.
    filename : str, optional
        Current script filename for figure directory naming. The default is None.
    **kwargs : dict
        Additional keyword arguments for customization.

    Returns
    -------
    params : dict
        Dictionary containing initialized parameters and objects for analysis.
        - ``TASK`` :            The task type.
        - ``MODEL_DIR`` :       Directory where models are stored.
        - ``LOAD_MODEL_DIR`` :  Path to the pretrained model file.
        - ``FIG_DIR`` :         Directory for saving figures.
        - ``N_IN`` :            Input size of the RNN.
        - ``N_HID`` :           Hidden size of the RNN.
        - ``N_OUT`` :           Output size of the RNN.
        - ``TOT_LEN`` :         Total length of the input sequences.
        - ``STIM_DUR`` :        Duration of the stimulus.
        - ``STIM_INTERVAL`` :   Interval between stimuli.
        - ``TARGET_LEN`` :      Length of the target output.
        - ``N_CLASS`` :         Number of classes in the task.
        - ``N_SAMPLES`` :       Number of samples / data to analyze.
        - ``PERMS`` :           List of permutations for class indices.
        - ``CLASS_INDEX`` :     Array of class indices.
        - ``VAL_LOADER`` :      DataLoader for validation data.
        - ``OUTPUTS_DICT`` :    Dictionary of RNN outputs and parameters.
        - ``TIMEPOINTS`` :      Timepoints of the input sequences.
        - ``RNN_CONFIG`` :      Configuration dictionary for the RNN.
    """
    model_dir = MODEL_BASE / "temp"
    load_model_dir = model_dir / f"{task}_latest.pth"
    model = torch.load(load_model_dir, map_location=DEVICE)

    print("[INFO] Load model from:", load_model_dir)

    config = model["config"]
    cfg = SynConfig(**config)

    # task variants
    if task == "forward2backward" or task == "layernorm_backward":
        # forward2backward: pretrained forward model, finetune to backward
        # layernorm_backward: backward task with layernorm during training
        task = "backward"

    # suppose the current filename is "subspace_{xxx}.py"
    # extract "xxx" from the filename
    if filename.startswith("subspace_") and filename.endswith(".py"):
        extracted_name = filename[len("subspace_") : -len(".py")]
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

    fig_dir = FIG_BASE / extracted_name
    Path.mkdir(fig_dir, parents=True, exist_ok=True)

    class_index = np.arange(0, n_samples)

    # parse kwargs
    noise_sigma = kwargs.get("noise_sigma", 0.0)
    single_stage_noise = kwargs.get("single_stage_noise", None)
    framework = kwargs.get("framework", DEFAULT_FRAMEWORK)

    rnn_config = dict(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        output_size=cfg.output_size,
        activation_fn="tanh",
        dt=cfg.dt,
        tau=cfg.tau,
        g=cfg.g,
        h0_trainable=False,
        use_layernorm=False,
    )

    val_dataset = abcDataset(
        noise_sigma=noise_sigma,
        tot_len=cfg.val_T,
        stim_dur=cfg.stim_dur,
        stim_interval=cfg.stim_interval,
        target_len=cfg.target_len,
        num_data=n_samples,
        class_index=class_index,
        task_mode=task,
        regress_target_mode=DEFAULT_REGRESS_MODE,
        signle_stage_noise=single_stage_noise,
        seed=cfg.seed,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=n_samples, shuffle=False, num_workers=0
    )

    outputs_dict = get_rnn_outputs(
        rnn_config, load_model_dir, val_loader, framework=framework
    )

    timepoints = val_dataset.get_timepoints()

    params = {
        "TASK": task,
        "MODEL_DIR": model_dir,
        "LOAD_MODEL_DIR": load_model_dir,
        "FIG_DIR": fig_dir,
        "N_IN": cfg.input_size,
        "N_HID": cfg.hidden_size,
        "N_OUT": cfg.output_size,
        "TOT_LEN": cfg.val_T,
        "STIM_DUR": cfg.stim_dur,
        "STIM_INTERVAL": cfg.stim_interval,
        "TARGET_LEN": cfg.target_len,
        "N_CLASS": DEFAULT_N_CLASS,
        "N_SAMPLES": n_samples,
        "PERMS": list(itertools.permutations(range(6), 3)),
        "CLASS_INDEX": class_index,
        "VAL_LOADER": val_loader,
        "OUTPUTS_DICT": outputs_dict,
        "TIMEPOINTS": timepoints,
        "RNN_CONFIG": rnn_config,
    }

    return params


def shade_color(rgb, factor=1.0):
    """Return a shaded color by converting to HLS and scaling lightness.
    factor < 1 -> darker, factor > 1 -> lighter (clamped).
    """
    r, g, b, a = rgb
    # colorsys uses RGB and HLS values in 0..1
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    # scale lightness
    lightness = max(0.0, min(1.0, lightness * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return (r2, g2, b2, a)


def get_color(X1, X2, X3):
    """
    Generate colors for rank 1, 2, 3 based on base colors and shading factors.

    Parameters
    ----------
    X1 : list
        List of item indices for rank 1.
    X2 : list
        List of item indices for rank 2.
    X3 : list
        List of item indices for rank 3.

    Returns
    -------
    colors_rank1 : list
        List of colors for rank 1 items.
    colors_rank2 : list
        List of colors for rank 2 items.
    colors_rank3 : list
        List of colors for rank 3 items.
    """
    # Create a base hue for each item and generate rank-wise shades (rank1 darkest -> rank3 lightest)
    # use tab10 as base for up to 10 items; we have 6 items
    base_cmap = plt.get_cmap("tab10")
    base_colors = [base_cmap(i) for i in range(6)]

    # choose factors so rank1 is darkest, rank2 medium, rank3 lightest
    rank_factors = {1: 0.8, 2: 1.05, 3: 1.4}

    colors_rank1 = [shade_color(base_colors[i], rank_factors[1]) for i in X1]
    colors_rank2 = [shade_color(base_colors[i], rank_factors[2]) for i in X2]
    colors_rank3 = [shade_color(base_colors[i], rank_factors[3]) for i in X3]

    return base_colors, colors_rank1, colors_rank2, colors_rank3


def get_color_single(X1, X2, X3):
    color_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]

    colors_rank1 = [color_list[i] for i in X1]
    colors_rank2 = [color_list[i] for i in X2]
    colors_rank3 = [color_list[i] for i in X3]

    return color_list, colors_rank1, colors_rank2, colors_rank3


#! ------------- PCA AND REGRESSION HELP FUNCTIONS -------------
def pca_single(
    hiddens: np.ndarray, n_components: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on per-sample mean activity and return per-class mean projections.

    Parameters
    ----------
    hiddens : ndarray, shape (n_samples, tot_len, N_HID)
        Hidden activity sequences.
    n_components : int, optional
        Number of PCA components to retain. Default is 8.

    Returns
    -------
    Low : ndarray, shape (n_classes, n_components)
        Per-class mean projection in PCA space.
    var_explained : ndarray
        Explained variance ratio for each PCA component.
    """
    n_samples, length, n_hid = hiddens.shape
    # use time-averaged activity per sample to fit PCA
    delay_mean = np.mean(hiddens, axis=1).reshape(-1, n_hid)

    pca_x = PCA(n_components=n_components).fit(delay_mean)
    # transform all timepoints and reshape back to (n_class, tot_len, n_components)
    Low = np.mean(
        pca_x.transform(hiddens.reshape(-1, n_hid)).reshape(
            n_samples, length, n_components
        ),
        axis=1,
    )
    var_explained = pca_x.explained_variance_ratio_
    return Low, var_explained, pca_x


def neo_compute_principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute principal angles (in degrees) between two subspaces.

    Parameters
    ----------
    U : ndarray, shape (d1, k1)
        Basis matrix for the first subspace (columns need not be orthonormal).
    V : ndarray, shape (d2, k2)
        Basis matrix for the second subspace (columns need not be orthonormal).

    Returns
    -------
    angles_deg : ndarray
        Principal angles in degrees (values in 0..90).
    """
    # orthonormalize columns via QR to improve numerical stability
    Q_U, _ = np.linalg.qr(U, mode="reduced")
    Q_V, _ = np.linalg.qr(V, mode="reduced")

    # cross-correlation between orthonormal bases
    M = Q_U.T @ Q_V

    # economy SVD; singular values ~ cos(theta)
    _, s, _ = np.linalg.svd(M, full_matrices=False)

    # ensure values fed to arccos are in [0,1] -> angles in [0,90] degrees
    s = np.clip(s, 0.0, 1.0)

    return np.rad2deg(np.arccos(s))


def compute_subspace_angles(
    h_layers: dict[str, np.ndarray],
) -> tuple[dict[str, dict[int, np.ndarray]], np.ndarray, list[str]]:
    """
    Compute mean principal angles between 2D subspaces across multiple layers.

    For each layer in ``h_layers`` the function fits PCA (6 components) and
    groups PCs into three 2D subspaces: (PC1-2), (PC3-4), (PC5-6). It returns a
    dictionary of subspace bases and a pairwise matrix of average principal
    angles (in degrees).

    Parameters
    ----------
    h_layers : dict
        Mapping layer name -> activations array of shape (n_samples, n_features).

    Returns
    -------
    subspace_bases : dict
        Mapping layer name -> dict(sub_id -> basis_matrix) where basis_matrix has
        shape (d, 2).
    angle_matrix : ndarray, shape (L*3, L*3)
        Pairwise matrix of mean principal angles (degrees) between all subspaces.
    labels : list of str
        Labels corresponding to rows/columns of the angle matrix.
    """
    # extract PCA-based 2D bases for each layer
    subspace_bases = {}
    layer_keys = list(h_layers.keys())
    for layer_name, H in h_layers.items():
        pca = PCA(n_components=6).fit(H)
        bases = {
            0: pca.components_[[0, 1]].T,  # PC1-2
            1: pca.components_[[2, 3]].T,  # PC3-4
            2: pca.components_[[4, 5]].T,  # PC5-6
        }
        subspace_bases[layer_name] = bases

    # build labels for all subspaces
    labels = [f"{k}_Sub{s+1}" for k in layer_keys for s in range(3)]

    # compute pairwise mean principal angles
    L = len(labels)
    angle_matrix = np.zeros((L, L))
    all_pairs = [(k, s) for k in h_layers for s in [0, 1, 2]]
    for i, (src_layer, src_sub) in enumerate(all_pairs):
        U = subspace_bases[src_layer][src_sub]
        for j in range(i, L):
            tgt_layer, tgt_sub = all_pairs[j]
            V = subspace_bases[tgt_layer][tgt_sub]
            svals = neo_compute_principal_angles(U, V)
            m = np.mean(svals)
            angle_matrix[i, j] = m
            angle_matrix[j, i] = m  # mirror

    return subspace_bases, angle_matrix, labels
