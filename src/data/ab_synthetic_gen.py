import itertools

import numpy as np
import torch
from torch.utils.data import Dataset

# fmt: off
#! ------------- TIME CONSTANTS (all in timesteps) -------------
# Task timeline structure:
#   stim_init -> stim1 -> interval -> stim2 -> interval -> stim3 -> interval
#   -> pre_cue -> cue -> post_cue -> retrieval -> [blank padding]

DEFAULT_STIM_DUR = 8        # Duration of each stimulus presentation
DEFAULT_STIM_INTERVAL = 8   # Interval between consecutive stimuli
DEFAULT_TARGET_LEN = 12     # Duration of each target waveform during retrieval

STIM_INIT = 8               # Trial start: initial fixation duration before stim1

# Pre-cue period: The actual pre-cue duration is (stim3_interval + PRE_CUE_LEN).
PRE_CUE_LEN_SINGLE = 12
PRE_CUE_LEN_SWITCH = 16

CUE_LEN_SWITCH = 12         # Cue duration for switch task (forward/backward)
CUE_LEN_SINGLE = 0          # Cue duration for non-switch tasks (no cue needed)

# Delay after cue, before retrieval starts
POST_CUE_LEN_SINGLE = 12
POST_CUE_LEN_SWITCH = 24

TOT_LEN = 200               # Total trial length (includes blank padding at end)
RATIO = 0.25                # Ratio for computing randomized interval lengths (training mode)

#! ------------- HARDCODED CONSTANTS -------------
# note: Here 24 channels encode 6 items in a circle. The size 24 is
# chosen simply for better training performance.
N_STIM = 24                 # 24d stimulus
N_IN_SINGLE = N_STIM + 1    # 24d stimulus + 1d fixation
N_IN_SWITCH = N_STIM + 3    # 24d stimulus + 1d fixation + 2d cue
N_OUT = 7                   # 6d item + 1d fixation


# fmt: on
#! ------------- CACHE ITEM TEMPLATES -------------
def _precompute_item_templates():
    """
    Precompute all six item templates for faster access.
    """
    sigma2 = 2
    size = N_STIM
    x = np.linspace(0, 2 * np.pi, size, endpoint=False)

    templates = np.zeros((6, size))
    for item_index in range(6):
        miu = item_index * np.pi / 3
        x_shifted = (x - miu + np.pi) % (2 * np.pi) - np.pi
        templates[item_index] = np.exp(-(x_shifted**2) / (2 * sigma2)) * 4

    return templates


ITEM_TEMPLATES = _precompute_item_templates()


#! ------------- MULTI-HELP FUNCTIONS -------------
def get_item(item_index) -> np.ndarray:
    if item_index not in range(6):
        raise ValueError("No such item index! Only six classes (0-5).")

    return ITEM_TEMPLATES[item_index]


def random_interval(num: int, ratio: float = 0.5) -> int:
    """
    Sample uniformly from ``(1-ratio)`` to ``(1+ratio)`` of ``num`` as integers (inclusive).
    Ensure at least 1.
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("ratio should be in [0,1]")

    min_post = max(1, int(np.floor(num * (1 - ratio))))
    max_post = max(1, int(np.ceil(num * (1 + ratio))))

    if max_post < min_post:
        max_post = min_post

    return int(np.random.randint(min_post, max_post + 1))


#! ------------- DATA GENERATOR FUNCTIONS -------------
def get_data(
    stim_dur: int = DEFAULT_STIM_DUR,
    stim_interval: int = DEFAULT_STIM_INTERVAL,
    target_len: int = DEFAULT_TARGET_LEN,
    tot_len: int = TOT_LEN,
    num_data: int = 2000,
    class_index: np.ndarray = None,
    type: str = "analysis",
    task_mode: str = "switch",
    seq_len: int | None = 3,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Generate synthetic dataset for the "switch" abc task.

    Parameters
    ----------
    stim_dur : int, optional
        The activation duration of each stimulus, by default ``DEFAULT_STIM_DUR``

    stim_interval : int, optional
        The refractory time of each stimulus, by default ``DEFAULT_STIM_INTERVAL``

    target_len : int, optional
        The activation duration of each target, by default ``DEFAULT_TARGET_LEN``

    tot_len : int, optional
        Length of a whole input, by default ``TOT_LEN``

    num_data : int, optional
        Number of training data, by default 2000

    class_index : np.ndarray, optional
        The specified class index array of data, by default None

    type : str, optional
        Type of the dataset, by default "analysis". If "analysis", use fixed intervals;
        if "training", use random intervals

    task_mode : str, optional
        Task mode, by default "switch". Options are:
        - "switch": randomly switch between forward and backward recall
        - "forward": always forward recall
        - "backward": always backward recall

    seq_len : int | None, optional
        Sequence length of each trial. If None or "mixed", randomly choose from 1,2,3.
        By default 3.

    seed : int | None, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    tuple of four numpy arrays: (train_x, train_y, train_t, seq_lengths)

    - train_x : np.ndarray, shape (num_train, tot_len, N_IN), dtype=float
        Input time-series for each trial. ``N_IN`` is ``N_IN_SWITCH`` for "switch" tasks
        (27, cue channels included) or ``N_IN_SINGLE`` for non-switch tasks (25).

    - train_y : np.ndarray, shape (num_train,), dtype=int64
        Class index encoding the trial configuration.
        For a fixed seq_len ``L`` there are ``n_perms = P(6, L)`` base permutations.

        If task_mode == "switch", valid class indices are in ``[0, 2*n_perms-1]``:
            - indices ``0..(n_perms-1)`` => forward trials (base permutation)
            - indices ``n_perms..(2*n_perms-1)`` => backward trials (base permutation)

        If task_mode != "switch", indices are in ``[0, n_perms-1]`` and direction is given by task_mode.

    - train_t : np.ndarray, shape (num_train, 7), dtype=int64
        Key timepoints (in timesteps) for each trial:

            0: ``t1``  - stimulus 1 start
            1: ``t2``  - stimulus 2 start
            2: ``t3``  - stimulus 3 start
            3: ``t4``  - cue (or delay) start
            4: ``t5``  - pre-cue length
            5: ``t6``  - post-cue length
            6: ``t7``  - retrieval (target1) start
            7: ``t8``  - target length

    - seq_lengths : np.ndarray, shape (num_train,), dtype=int64
        Actual sequence length ``L`` (1,2,3) used for each trial. Useful when ``seq_len`` argument
        is None or "mixed" and lengths vary per trial.

    Notes
    -----
    - The total period timeaxis:

        | STIM_INIT | stim_dur+stim_interval | stim_dur+stim_interval |
        0           stim1                    stim2                    stim3

        | stim_dur+stim_interval | pre_cue_len | cue_len | post_cue_len |
        stim3                                  cue_st    cue_ed         target1

        | target_len | target_len | target_len |
        target1      target2      target3      end

    - If task_mode is "switch", the above period will be used. If task_mode is "forward"
    or "backward", the ``cue_len`` period will be blank, thus you may regard the
    ``pre_cue_len + cue_len + post_cue_len`` period as a single delay period.
    - In experiemnt settings, the delay period instantly follows the offset of stimulus 3,
    however, here we still add a stimulus interval after stimulus 3 for consistency.
    """

    # initialize RNG for reproducibility
    RNG = np.random.default_rng(seed)

    # prepare permutation lists for each sequence length (1,2,3)
    perms_by_len = {L: list(itertools.permutations(range(6), L)) for L in (1, 2, 3)}

    # normalize seq_len parameter: int (fixed), None or "mixed" (sample from 1..3)
    if seq_len is None or seq_len == "mixed":
        fixed_length = None
        allowed_lengths = [1, 2, 3]
        # probability weights for lengths 1,2,3 respectively
        allowed_probs = [0, 1 / 5.0, 4 / 5.0]
    elif isinstance(seq_len, int):
        if seq_len not in (1, 2, 3):
            raise ValueError("seq_len must be 1,2,3 or None/'mixed'")
        fixed_length = seq_len
        allowed_lengths = [seq_len]
    else:
        raise ValueError("seq_len must be 1,2,3, None or 'mixed'")

    N_IN = N_IN_SWITCH if task_mode == "switch" else N_IN_SINGLE
    CUE_LEN = CUE_LEN_SWITCH if task_mode == "switch" else CUE_LEN_SINGLE
    POST_CUE_LEN = POST_CUE_LEN_SWITCH if task_mode == "switch" else POST_CUE_LEN_SINGLE
    PRE_CUE_LEN = PRE_CUE_LEN_SWITCH if task_mode == "switch" else PRE_CUE_LEN_SINGLE

    train_x = np.zeros((num_data, tot_len, N_IN))
    train_y = np.zeros(num_data, np.int64)
    train_t = np.zeros((num_data, 8), np.int64)
    seq_lengths = np.zeros(num_data, np.int64)

    for i in range(num_data):
        # decide sequence length for this trial
        if fixed_length is not None:
            L = fixed_length
        else:
            # probabilistic choice with weights
            L = int(RNG.choice(allowed_lengths, p=allowed_probs))

        # timepoints
        if type == "training":
            stim1 = STIM_INIT + int(RNG.integers(0, STIM_INIT))
            stim2 = stim1 + stim_dur + int(RNG.integers(0, stim_interval))
            stim3 = stim2 + stim_dur + int(RNG.integers(0, stim_interval))
            stim4 = stim3 + stim_dur + int(RNG.integers(0, stim_interval))
            stim5 = random_interval(PRE_CUE_LEN, ratio=RATIO)
            stim6 = random_interval(POST_CUE_LEN, ratio=RATIO)
            stim7 = stim4 + stim5 + CUE_LEN + stim6
            stim8 = target_len
            cue_st = stim4 + stim5
        else:
            stim1 = STIM_INIT  # start of stim1
            stim2 = stim1 + stim_dur + stim_interval  # start of stim2
            stim3 = stim2 + stim_dur + stim_interval  # start of stim3
            stim4 = stim3 + stim_dur + stim_interval  # start of cue_pre or delay
            stim5 = PRE_CUE_LEN  # length of pre_cue
            stim6 = POST_CUE_LEN  # length of post_cue
            stim7 = stim4 + stim5 + CUE_LEN + stim6  # start of retrieval
            stim8 = target_len  # length of each target
            cue_st = stim4 + stim5  # start of cue

        # choose a permutation index for this trial
        perms = perms_by_len[L]
        n_perms = len(perms)

        # determine sampling / validation range depending on task_mode
        if class_index is None:
            if task_mode == "switch":
                perm_idx = RNG.integers(0, n_perms * 2)
            else:
                perm_idx = RNG.integers(0, n_perms)
        else:
            ci_arr = np.asarray(class_index)
            perm_idx = ci_arr[i]
            max_idx = n_perms * 2 if task_mode == "switch" else n_perms
            if perm_idx < 0 or perm_idx >= max_idx:
                raise IndexError(
                    f"class_index[{i}] = {perm_idx} out of range for seq_len={L}"
                )

        # determine forward/backward behavior using task_mode
        if task_mode == "switch":
            if perm_idx >= n_perms:
                is_backward = True
                items = perms[perm_idx - n_perms]
            else:
                is_backward = False
                items = perms[perm_idx]
        elif task_mode == "forward":
            is_backward = False
            items = perms[perm_idx % n_perms]
        elif task_mode == "backward":
            is_backward = True
            items = perms[perm_idx % n_perms]
        else:
            raise ValueError("task_mode must be one of 'switch','forward','backward'")

        # build stimulus values (up to 3 items)
        stim_values = [np.zeros(N_STIM), np.zeros(N_STIM), np.zeros(N_STIM)]

        for k, it in enumerate(items):
            # if k == 1:
            #     continue
            stim_values[k] = get_item(it)

        stim_value_1, stim_value_2, stim_value_3 = stim_values
        # (value_4, value_5) = (0, 1) -> forward task; (1, 0) -> backward task
        stim_value_4 = 1 if is_backward else 0
        stim_value_5 = 0 if is_backward else 1

        # write stimuli (only present those according to L)
        if L >= 1:
            train_x[i, stim1 : stim1 + stim_dur, 0:N_STIM] = stim_value_1.reshape(
                1, N_STIM
            )
        if L >= 2:
            train_x[i, stim2 : stim2 + stim_dur, 0:N_STIM] = stim_value_2.reshape(
                1, N_STIM
            )
        if L >= 3:
            train_x[i, stim3 : stim3 + stim_dur, 0:N_STIM] = stim_value_3.reshape(
                1, N_STIM
            )

        # cue channels
        if task_mode == "switch":
            train_x[i, cue_st : cue_st + CUE_LEN_SWITCH, N_STIM] = stim_value_4
            train_x[i, cue_st : cue_st + CUE_LEN_SWITCH, N_STIM + 1] = stim_value_5

        # fixation
        train_x[i, 0:stim7, -1] = 1

        train_t[i, 0] = int(stim1)  # start of stim1
        train_t[i, 1] = int(stim2)  # start of stim2
        train_t[i, 2] = int(stim3)  # start of stim3
        train_t[i, 3] = int(stim4)  # start of cue_pre or delay
        train_t[i, 4] = int(stim5)  # length of pre_cue
        train_t[i, 5] = int(stim6)  # length of post_cue
        train_t[i, 6] = int(stim7)  # start of retrieval
        train_t[i, 7] = int(stim8)  # length of each target

        # store perm index as class label; caller can use seq_lengths to decode
        train_y[i] = int(perm_idx)
        seq_lengths[i] = int(L)

    return train_x, train_y, train_t, seq_lengths


def get_regress_target_long(
    class_i, train_t, tot_len, seq_len: int = 3, task_mode: str = "switch"
):
    r"""
    Build regression target for retrieval.

    Parameters
    ----------
    class_i : int
    train_t : np.ndarray
        Timepoints array of shape ``(8,)``
    tot_len : int
    seq_len : int, optional
        Sequence length of this trial, by default 3
    task_mode : str, optional
        Task mode, by default "switch". Options are:
            - "switch": randomly switch between forward and backward recall
            - "forward": always forward recall
            - "backward": always backward recall

    Returns
    -------
    reg_out : np.ndarray
        Regression target of shape ``(tot_len, N_OUT)``
    """
    reg_out = np.zeros((tot_len, N_OUT))
    retr_start, target_len = train_t[6], train_t[7]

    perms = list(itertools.permutations(range(6), int(seq_len)))
    n_perms = len(perms)

    ci = int(class_i)
    # interpret class index according to task_mode
    if task_mode == "switch":
        # allow ci in 0...2*n_perms-1; >= n_perms indicates backward
        if ci >= n_perms:
            base_idx = ci - n_perms
            is_backward = True
        else:
            base_idx = ci
            is_backward = False
    elif task_mode == "backward":
        base_idx = ci % n_perms
        is_backward = True
    else:  # forward or other
        base_idx = ci % n_perms
        is_backward = False

    items = perms[base_idx]
    order = list(items[::-1]) if is_backward else list(items)

    for idx_item, item in enumerate(order):
        start = retr_start + idx_item * target_len
        end = start + target_len
        reg_out[start:end, item] = 1

    reg_out[:retr_start, -1] = 1
    return reg_out


#! ------------- DATASET CLASS -------------
class abcDataset(Dataset):
    r"""
    Dataset for synthetic ABC tasks (``forward``, ``backward``, ``switch``).

    Wraps the procedural generator :pyfunc:`get_data` and provides a PyTorch
    ``Dataset`` interface returning a single trial as a tuple
    ``(inputs, class_idx, timepoints, regress_target, seq_len)``.

    Parameters
    ----------
    noise_sigma : float, optional
        Standard deviation of additive Gaussian noise applied to inputs (default 0.05).

    tot_len : int, optional
        Total length (timesteps) of each trial (default ``TOT_LEN``).

    stim_dur : int, optional
        Stimulus presentation duration in timesteps (default ``DEFAULT_STIM_DUR``).

    stim_interval : int, optional
        Interval between stimuli in timesteps (default ``DEFAULT_STIM_INTERVAL``).

    target_len : int, optional
        Length of each retrieval/target window in timesteps (default ``DEFAULT_TARGET_LEN``).

    num_data : int, optional
        Number of samples to generate (default 2000).

    class_index : list or np.ndarray, optional
        Optional array of class indices to force specific permutations per sample.
        If omitted, class indices are sampled uniformly according to ``seq_len``.

    regress_target_mode : str, optional
        Mode used to build regression targets. Supported values:
        - ``"hierarchy_long"`` (default): structured retrieval targets over time.
        - ``"constant"``: deprecated in this codebase.

    type : {"analysis","training"}, optional
        If ``"analysis"``, use fixed, deterministic timing; if ``"training"``, use
        randomized intervals for data augmentation (default ``"analysis"``).

    task_mode : {"switch","forward","backward"}, optional
        Task variant controlling cue channels and recall direction (default ``"switch"``).

    seq_len : int, None or "mixed", optional
        Sequence length used when generating samples. If an ``int`` (1/2/3), that
        length is fixed for all trials. If ``None`` or ``"mixed"``, sequence lengths
        are sampled from {1,2,3} per trial (default 3).

    seed : int, optional
        Random seed for reproducibility.

    signle_stage_noise : dict, optional
        If provided, applies extra additive noise to a single stage.
        Expected keys: ``{"stage": "stim1"|"stim2"|"stim3", "noise_level": float}``.

    Returns
    -------
    The dataset yields tuples per item:

        (inputs, class_idx_tensor, timepoints_tensor, regress_target, seq_len_tensor)

    - ``inputs``: ``torch.FloatTensor`` of shape ``(tot_len, input_dim)``
    - ``class_idx_tensor``: ``torch.LongTensor`` (scalar) encoding the permutation index
    - ``timepoints_tensor``: ``torch.LongTensor`` of shape ``(8,)`` with key timepoints
    - ``regress_target``: ``torch.FloatTensor`` of shape ``(tot_len, N_OUT)``
    - ``seq_len_tensor``: ``torch.LongTensor`` (scalar) equal to 1/2/3 for that trial

    Notes
    -----
    - When ``seq_len`` is ``None`` or ``"mixed"``, the returned ``seq_len_tensor``
    records the actual length used for each sample.
    """

    def __init__(
        self,
        noise_sigma=0.05,
        tot_len=TOT_LEN,
        stim_dur=DEFAULT_STIM_DUR,
        stim_interval=DEFAULT_STIM_INTERVAL,
        target_len=DEFAULT_TARGET_LEN,
        num_data=2000,
        class_index=None,
        regress_target_mode="hierarchy_long",
        type="analysis",  # "analysis" or "training" (for potential data augmentation in training phase)
        task_mode: str = "switch",
        seq_len: int | None = 3,
        seed: int | None = None,
        signle_stage_noise: dict = None,  # only apply noise to one stage
    ):

        self.noise_sigma = noise_sigma
        self.regress_target_mode = regress_target_mode
        self.stim_dur = stim_dur
        self.stim_interval = stim_interval
        self.target_len = target_len
        self.tot_len = tot_len
        self.type = type
        self.task_mode = task_mode
        self.seed = seed
        self.noise_sigma = noise_sigma
        self.regress_target_mode = regress_target_mode
        self.single_stage_noise = signle_stage_noise

        # call get_data and retrieve seq_lens
        self.data_x, self.data_y, self.data_t, self.seq_lens = get_data(
            stim_dur=stim_dur,
            stim_interval=stim_interval,
            target_len=target_len,
            tot_len=tot_len,
            num_data=num_data,
            class_index=class_index,
            type=type,
            task_mode=task_mode,
            seq_len=seq_len,
            seed=seed,
        )

    def __getitem__(self, index):
        data_x_i = self.data_x[index].copy()
        data_y_i = self.data_y[index].copy()
        data_t_i = self.data_t[index].copy()
        deta_seq_len_i = self.seq_lens[index].copy()

        if self.regress_target_mode == "hierarchy_long":
            data_reg_i = get_regress_target_long(
                data_y_i,
                data_t_i,
                self.tot_len,
                seq_len=deta_seq_len_i,
                task_mode=self.task_mode,
            )

        elif self.regress_target_mode == "constant":
            raise ValueError("Deprecated regress_target_mode!")
            data_reg_i = np.zeros((data_x_i.shape[0], 10))
            data_reg_i[:, data_y_i] = 1.0

        else:
            raise ValueError("Unsupported regress_target_mode!")

        # apply global noise, fixation channel included
        if self.noise_sigma > 0:
            data_x_i += self.noise_sigma * np.random.normal(0, 1, size=data_x_i.shape)

        # apple stage-specific noise
        if self.single_stage_noise is not None:
            stage = self.single_stage_noise.get("stage", None)
            noise_level = self.single_stage_noise.get("noise_level", 0.0)
            if stage is not None and noise_level > 0.0:
                t1, t2, t3 = data_t_i[:3]
                if stage == "stim1":
                    start_t = t1
                    end_t = start_t + self.stim_dur
                elif stage == "stim2":
                    start_t = t2
                    end_t = start_t + self.stim_dur
                elif stage == "stim3":
                    start_t = t3
                    end_t = start_t + self.stim_dur
                else:
                    raise ValueError("Unsupported stage for single_stage_noise!")

                data_x_i[start_t:end_t, 0:N_STIM] += noise_level * np.random.normal(
                    0, 1, size=data_x_i[start_t:end_t, 0:N_STIM].shape
                )

        return (
            torch.FloatTensor(data_x_i),
            torch.LongTensor([data_y_i]),
            torch.LongTensor(data_t_i),
            torch.FloatTensor(data_reg_i),
            torch.LongTensor([deta_seq_len_i]),
        )

    def __len__(self):
        return len(self.data_y)

    def get_timepoints(self) -> dict:
        """
        Return a dictionary of key timepoints for the trial structure. The
        timepoints are based on the analysis mode with **fixed intervals**.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - ``stim1_start`` -> Start of stimulus 1
            - ``stim2_start`` -> Start of stimulus 2
            - ``stim3_start`` -> Start of stimulus 3
            - ``stim1_off`` -> Offset of stimulus 1
            - ``stim2_off`` -> Offset of stimulus 2
            - ``stim3_off`` -> Offset of stimulus 3
            - ``stim1_end`` -> End of stimulus 1 (including interval)
            - ``stim2_end`` -> End of stimulus 2 (including interval)
            - ``stim3_end`` -> End of stimulus 3 (including interval)
            - ``pre_cue_start`` -> Start of pre-cue period
            - ``cue_start`` -> Start of cue period
            - ``cue_end`` -> End of cue period
            - ``post_cue_end`` -> End of post-cue period
            - ``delay_start`` -> Start of delay period
            - ``delay_end`` -> End of delay period
            - ``target1_start`` -> Start of target 1 output period
            - ``target2_start`` -> Start of target 2 output period
            - ``target3_start`` -> Start of target 3 output period
            - ``target1_end`` -> End of target 1 output period
            - ``target2_end`` -> End of target 2 output period
            - ``target3_end`` -> End of target 3 output period
            - ``end`` -> End of the final target

        Notes
        -----
        - We provide the returned dict for the **compatibility** of both switch
        and non-switch tasks. For non-switch tasks, the ``delay_start`` and
        ``delay_end`` correspond to the whole ``pre_cue_len + cue_len + post_cue_len``
        period.
        """
        STIM_PERIOD = self.stim_dur + self.stim_interval
        CUE_LEN = CUE_LEN_SWITCH if self.task_mode == "switch" else CUE_LEN_SINGLE
        POST_CUE_LEN = (
            POST_CUE_LEN_SWITCH if self.task_mode == "switch" else POST_CUE_LEN_SINGLE
        )
        PRE_CUE_LEN = (
            PRE_CUE_LEN_SWITCH if self.task_mode == "switch" else PRE_CUE_LEN_SINGLE
        )

        # the start timepoint of each stimulus
        stim1_start = STIM_INIT
        stim2_start = STIM_INIT + STIM_PERIOD
        stim3_start = stim2_start + STIM_PERIOD

        # the offset timepoint of each stimulus
        stim1_off = stim1_start + self.stim_dur
        stim2_off = stim2_start + self.stim_dur
        stim3_off = stim3_start + self.stim_dur

        # the end timepoint of each stimulus (total period, including interval)
        stim1_end = stim1_start + STIM_PERIOD
        stim2_end = stim2_start + STIM_PERIOD
        stim3_end = stim3_start + STIM_PERIOD

        # cue period
        # pre_cue_start = stim3_start + self.stim_dur
        # cue_start = pre_cue_start + self.stim_interval + PRE_CUE_LEN
        pre_cue_start = stim3_start + STIM_PERIOD
        cue_start = pre_cue_start + PRE_CUE_LEN
        cue_end = cue_start + CUE_LEN
        post_cue_end = cue_end + POST_CUE_LEN

        # delay period
        # (for non-switch tasks, this is the whole pre_cue+cue+post_cue period)
        delay_start = pre_cue_start
        delay_end = post_cue_end

        # target start timepoints
        target1_start = cue_end + POST_CUE_LEN
        target2_start = target1_start + DEFAULT_TARGET_LEN
        target3_start = target2_start + DEFAULT_TARGET_LEN

        # target end timepoints
        target1_end = target1_start + DEFAULT_TARGET_LEN
        target2_end = target2_start + DEFAULT_TARGET_LEN
        target3_end = target3_start + DEFAULT_TARGET_LEN

        timepoints: dict = {
            # the start timepoint of each stimulus
            "stim1_start": stim1_start,
            "stim2_start": stim2_start,
            "stim3_start": stim3_start,
            # the offset timepoint of each stimulus
            "stim1_off": stim1_off,
            "stim2_off": stim2_off,
            "stim3_off": stim3_off,
            # the end timepoint of each stimulus (total period, including interval)
            "stim1_end": stim1_end,
            "stim2_end": stim2_end,
            "stim3_end": stim3_end,
            # cue period
            "pre_cue_start": pre_cue_start,
            "cue_start": cue_start,
            "cue_end": cue_end,
            "post_cue_end": post_cue_end,
            # delay period
            # (for non-switch tasks, this is the whole pre_cue+cue+post_cue period)
            "delay_start": delay_start,
            "delay_end": delay_end,
            # target start timepoints
            "target1_start": target1_start,
            "target2_start": target2_start,
            "target3_start": target3_start,
            # target end timepoints
            "target1_end": target1_end,
            "target2_end": target2_end,
            "target3_end": target3_end,
            "end": self.tot_len,
        }
        return timepoints


#! ------------- DATA VISUALIZATION -------------
if __name__ == "__main__":
    """
    CLI-capable visualization for task variants.

    Usage (from command line):
        python synthetic_ab_gen_switch.py [variant]

    Where variant is one of: switch-forward, switch-backward, forward, backward.
    If no variant is provided, all four are plotted together.
    """
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

    import logging
    import math

    import matplotlib.pyplot as plt

    import lib.utils as utils

    ## visualize all possible data
    # plt.figure(figsize=(8, 5))
    # for idx in range(6):
    #     y = get_item(idx)
    #     plt.plot(y, label=f"item {idx+1}")
    # plt.title("all possible results of the input gaussian curves")
    # plt.xlabel("input dimension (1~24)")
    # plt.ylabel("activity")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.show()
    ## visualize a single trial
    TYPE = "analysis"  # "analysis" or "training"
    base_idx = 3

    LEVEL = logging.DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger = utils.get_logger(LEVEL)
    logger.info(f"Using logging level: {logging.getLevelName(LEVEL)}")

    def resolve_variant(variant: str, base_idx: int, seq_len: int = 3):
        """Return (task_mode, class_idx) normalized for given seq_len."""
        n_perms_local = math.perm(6, seq_len)
        base = int(base_idx) % n_perms_local
        if variant == "switch-forward":
            return "switch", base
        if variant == "switch-backward":
            return "switch", base + n_perms_local
        if variant == "forward":
            return "forward", base
        if variant == "backward":
            return "backward", base
        raise ValueError(f"Unknown variant: {variant}")

    def plot_trial(ax, x, r, t, seq_len=3, stim_dur=8, variant=None, **kwargs):
        logger.debug(f"Plotting trial with variant={variant}, seq_len={seq_len}")

        x_arr = np.asarray(x)
        r_arr = np.asarray(r)
        t_vals = np.asarray(t)

        # t1: stimulus 1 start, t2: stimulus 2 start, t3: stimulus 3 start
        # t4: pre_cue start, t5: pre cue length, t6: post cue length
        # t7: retrieval start, t8: target length
        t1, t2, t3, t4, t5, t6, t7, t8 = [int(v) for v in t_vals]

        # cue timepoints, only for switch task
        cue_start = t4 + t5
        cue_end = cue_start + CUE_LEN_SWITCH

        # detect presented items
        presented_items = utils.decode_presented_items(x, t, stim_dur, seq_len)

        # detect retrieval outputs
        target_items = []
        for idx in range(seq_len):
            start = t7 + idx * t8
            end = start + t8
            out_win = r_arr[start:end, :6]

            target_items.append(int(np.argmax(np.mean(out_win, axis=0))))

        # infer backward if cue channels present or mode explicitly set to backward
        has_cue = x_arr.shape[1] == (N_STIM + 3)
        logger.debug(
            f"Presented items: {presented_items} | Target items: {target_items}"
        )

        inferred_backward = target_items == presented_items[::-1]

        input_color = "cyan" if inferred_backward else "blue"
        output_color = "magenta" if inferred_backward else "red"

        for i in range(x_arr.shape[1]):
            ax.plot(
                x_arr[:, i],
                color=input_color,
                alpha=0.25,
                label=(
                    "Input (backward)"
                    if inferred_backward and i == 0
                    else (
                        "Input (forward)" if not inferred_backward and i == 0 else None
                    )
                ),
            )
        for j in range(r_arr.shape[1]):
            ax.plot(
                r_arr[:, j],
                color=output_color,
                alpha=0.4,
                label=(
                    "Output (backward)"
                    if inferred_backward and j == 0
                    else (
                        "Output (forward)" if not inferred_backward and j == 0 else None
                    )
                ),
            )

        display_order = [1, 2, 3] if not inferred_backward else [3, 2, 1]
        tgt_colors = (
            ["green", "orange", "purple"]
            if not inferred_backward
            else ["violet", "gold", "lime"]
        )
        for idx, itm in enumerate(target_items):
            start = t7 + idx * t8
            end = start + t8 - 1
            label = f"item{display_order[idx]} (idx={itm})"
            ax.axvspan(start, end, color=tgt_colors[idx], alpha=0.15, label=label)
            ylim = ax.get_ylim()
            y_text = ylim[1] - 0.05 * (ylim[1] - ylim[0])
            ax.text(
                (start + end) / 2,
                y_text,
                str(itm),
                ha="center",
                va="top",
                fontsize=9,
                color="k",
            )

        # stimulus and interval
        stim_colors = ["deepskyblue", "dodgerblue", "navy"]
        interval_colors = ["lightblue", "skyblue", "midnightblue"]

        for i in range(seq_len):
            stim_start = [t1, t2, t3][i]
            stim_end = stim_start + stim_dur - 1
            interval_start = stim_end + 1
            interval_end = [t2, t3, t4][i] - 1

            ax.axvspan(
                stim_start,
                stim_end,
                color=stim_colors[i],
                alpha=0.25,
                label=f"stimulus {i + 1}",
            )
            ax.axvspan(
                interval_start,
                interval_end,
                color=interval_colors[i],
                alpha=0.10,
                label=f"interval {i + 1}",
            )

        # cue or delay
        if has_cue:
            ax.axvspan(t4, cue_start - 1, color="gray", alpha=0.05, label="pre cue")
            ax.axvspan(cue_start, cue_end - 1, color="gray", alpha=0.2, label="cue")
            ax.axvspan(cue_end, t7 - 1, color="gray", alpha=0.05, label="post cue")
        else:
            ax.axvspan(t4, t7 - 1, color="gray", alpha=0.05, label="delay")

        ax.set_ylabel("Value")
        ax.set_xlabel("Time")

        enable_legend = kwargs.get("enable_legend", True)
        no_border = kwargs.get("no_border", False)

        if enable_legend:
            # tidy legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = {}
            ordered = []
            for h, lbl in zip(handles, labels):
                if lbl and lbl not in by_label:
                    by_label[lbl] = h
                    ordered.append(lbl)
            if ordered:
                ax.legend(
                    [by_label[lbl] for lbl in ordered], ordered, loc="upper right"
                )

        if no_border:
            # Hide all axes, ticks and labels so only the plotted content remains.
            # Use set_axis_off() which hides spines, ticks and labels but preserves the
            # Artists (lines, patches) already drawn on the axes.
            ax.set_axis_off()

    def visualize_variant(
        variant: str, base_idx: int = 5, show: bool = True, seq_len: int = 3
    ):
        task_mode, class_idx = resolve_variant(variant, base_idx, seq_len)

        ds = abcDataset(
            noise_sigma=0.0,
            num_data=1,
            class_index=[class_idx],
            type=TYPE,
            task_mode=task_mode,
            seq_len=seq_len,
        )

        x, y, t, r, s = ds[0]
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plot_trial(
            ax, x, r, t, seq_len=seq_len, stim_dur=DEFAULT_STIM_DUR, variant=variant
        )
        ax.set_title(
            f"Variant={variant}  (task_mode={task_mode}, class_index={class_idx}, seq_len={seq_len})"
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def visualize_variants_grid(
        base_idx: int = 5,
        variants=None,
        ncols: int = 2,
        figsize=(14, 8),
        show: bool = True,
        save_path: str | None = None,
        seq_len: int = 3,
    ):
        if variants is None:
            variants = ["switch-forward", "switch-backward", "forward", "backward"]
        n = len(variants)
        ncols = max(1, ncols)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # turn off any extra axes
        for ax in axes_flat[n:]:
            ax.axis("off")

        for idx, variant in enumerate(variants):
            task_mode, class_idx = resolve_variant(variant, base_idx, seq_len)

            dataset = abcDataset(
                noise_sigma=0.0,
                num_data=1,
                class_index=[class_idx],
                type=TYPE,
                task_mode=task_mode,
                seq_len=seq_len,
            )

            x, y, t, r, s = dataset[0]
            ax = axes_flat[idx]
            plot_trial(
                ax, x, r, t, seq_len=seq_len, stim_dur=DEFAULT_STIM_DUR, variant=variant
            )
            ax.set_title(
                f"Variant={variant}  (task_mode={task_mode}, class_index={class_idx})"
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig, axes

    # run
    if len(sys.argv) > 1:
        visualize_variant(sys.argv[1], base_idx=base_idx, seq_len=3)
    else:
        # for var in ["switch-forward", "switch-backward", "forward", "backward"]:
        #     visualize_variant(var, base_idx=base_idx, seq_len=3)

        visualize_variants_grid(base_idx=base_idx, ncols=2, seq_len=3)
