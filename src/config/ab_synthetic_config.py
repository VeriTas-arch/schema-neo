from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch


@dataclass
class SynConfig:
    """
    Configuration class for PyTorch training.
    """

    # supplementary information
    description: str = "new GPU techniques test"

    # identity settings
    task: str = "base"

    # device settings
    cuda: bool = field(default_factory=lambda: torch.cuda.is_available())
    batch_size: int = 16  #! batch size == 256 isn't good
    # 由于 seq_len=3 时，总共只有 120 种组合，因此 batch_size 过大会导致一个 batch 内就
    # 包含所有类别，模型得到了全局信息，无法进行有效训练

    # basic directories, will be set in __post_init__
    model_dir: Path = None
    work_dir: Path = None

    # the project root directory is the parent of `src` folder
    project_root_dir: ClassVar[Path] = Path(__file__).parent.parent.parent

    # load pretrained model settings
    load_pretrained_model: bool = False
    load_pretrained_dir: Path = None
    trainable_layers: list = None  #! for compatibility only, use freeze_except instead
    freeze_except: list = None

    # optimizer and learning rate settings
    optimizer: dict = field(
        default_factory=lambda: dict(
            lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True
        )
    )

    # runtime settings
    data_workers: int = field(
        default_factory=lambda: 24 if torch.cuda.is_available() else 0
    )
    total_epochs: int = -1  # will be set in __post_init__ after task is known
    max_grad_norm: float = 0.2

    # model parameters
    model: str = "SimpleRNN"

    input_size: int = -1  # will be set in __post_init__ after task is known
    hidden_size: int = 400  # number of neurons in RNN 需要大不少
    output_size: int = 7

    activation_fn: str = "tanh"

    g: float = 1.0
    tau: float = 1.0
    dt: float = 0.05

    h0_trainable: bool = False

    # allow callers to pass a pre-built rnn_config dict; otherwise fill defaults in __post_init__
    rnn_config: dict = field(default_factory=dict)

    # dataset parameters
    seed: int = np.random.randint(1000, 2000)

    stim_dur: int = 8
    stim_interval: int = 8
    target_len: int = 12

    train_T: int = 200
    val_T: int = 200
    input_noise_sigma: float = 0.01

    train_data: int = 3000
    val_data: int = 300
    regress_target_mode: str = "hierarchy_long"

    # Logging settings
    log_level: str = "INFO"

    def __post_init__(self):
        #! ----- TASK DEPENDENT VARIABLES -----
        self.input_size: int = 27 if self.task == "switch" else 25
        self.total_epochs: int = 360 if self.task == "switch" else 360

        if self.model_dir is None and self.work_dir is None:
            # Model directory at project root (next to 'src')
            ext_name = f"{datetime.now():%Y%m%d_%H%M%S}"
            self.model_dir = self.project_root_dir / "model" / f"{self.task}"
            self.work_dir = f"{self.model_dir}/{ext_name}"

        #! ----- PRETRAINED MODEL LOADING -----
        self.load_pretrained_dir = (
            self.project_root_dir / "model" / "temp" / "forward_latest.pth"
        )
        self.freeze_except = ["Wr", "bias"]

        # print("[DEBUG] batch resize factor:", (1 + self.batch_size / 256))
        # self.optimizer["lr"] = min(2e-3, 1e-3 * (1 + self.batch_size / 128))

        # populate rnn_config only if caller did not provide one
        if not isinstance(self.rnn_config, dict) or len(self.rnn_config) == 0:
            self.rnn_config = dict(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                activation_fn=self.activation_fn,
                g=self.g,
                dt=self.dt,
                tau=self.tau,
                h0_trainable=self.h0_trainable,
            )

    def to_dict(self):
        """Convert configuration to dictionary format for display.

        - Path objects are converted to strings.
        """

        config_dict = self.__dict__.copy()
        for key, value in config_dict.items():
            # simple Path -> string
            if isinstance(value, Path):
                config_dict[key] = str(value)
        return config_dict
