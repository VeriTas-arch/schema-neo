import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser

import torch

# from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import lib.model as network
from config import SynConfig
from data import abcDataset
from hook import TrainingHook
from lib import utils

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"


def parse_args():
    parser = ArgumentParser(description="Train ABC delayed sequence-reproduction task")
    parser.add_argument(
        "--task",
        type=str,
        choices=["forward", "backward", "switch"],
        required=True,
        help="task type: forward/backward/switch",
    )

    return parser.parse_args()


def main():
    # set deterministic behavior
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if CUDA:
        torch.cuda.empty_cache()

    # verify task type
    args = parse_args()
    TASK = args.task

    # load config according to task type
    cfg = SynConfig(task=TASK)

    logger = utils.get_logger(cfg.log_level)
    logger.info(f"Task type: {TASK}")

    utils.set_seed(cfg.seed)

    train_dataset = abcDataset(
        noise_sigma=cfg.input_noise_sigma,
        tot_len=cfg.train_T,
        stim_dur=cfg.stim_dur,
        stim_interval=cfg.stim_interval,
        target_len=cfg.target_len,
        num_data=cfg.train_data,
        class_index=None,
        regress_target_mode=cfg.regress_target_mode,
        type="training",
        task_mode=TASK,
        seq_len="mix",
        seed=cfg.seed,
    )

    val_dataset = abcDataset(
        noise_sigma=cfg.input_noise_sigma,
        tot_len=cfg.val_T,
        stim_dur=cfg.stim_dur,
        stim_interval=cfg.stim_interval,
        target_len=cfg.target_len,
        num_data=cfg.val_data,
        class_index=None,
        regress_target_mode=cfg.regress_target_mode,
        type="analysis",
        task_mode=TASK,
        seq_len="mix",
        seed=cfg.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.data_workers,
        pin_memory=CUDA,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data_workers,
        pin_memory=CUDA,
    )

    model: torch.nn.Module = getattr(network, cfg.model)(**cfg.rnn_config)

    if cfg.load_pretrained_model and cfg.load_pretrained_dir is not None:
        model = utils.load_pretrained_model(
            model, cfg.load_pretrained_dir, freeze_except=cfg.freeze_except
        )
        logger.info(f"Loaded pretrained model from {cfg.load_pretrained_dir}")
        logger.info(f"Trainable layers: {cfg.freeze_except}")

    if CUDA:
        logger.info("============= Using CUDA =============")

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer)
    max_grad_norm = getattr(cfg, "max_grad_norm", 5.0)
    # scheduler = MultiStepLR(optimizer, milestones=[180], gamma=0.5)

    # Initialize TrainingHook
    training_hook = TrainingHook(
        save_dir=cfg.work_dir,
        save_interval=10,
        visualization_interval=10,
        log_interval=int(cfg.train_data / cfg.batch_size / 5),
        max_samples=2,
        enable_checkpoint=True,
        enable_visualization=True,
        enable_logging=True,
        logger=logger,
    )

    # Training loop
    for epoch in range(cfg.total_epochs):
        training_hook.before_epoch(epoch, cfg)

        model.train()
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            output = utils.batch_processor(model, data, cfg)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            training_hook.after_batch(batch_idx, output, model)

        # scheduler.step()
        training_hook.after_epoch(epoch, model, optimizer, val_loader, cfg)

    training_hook.after_run(cfg.total_epochs, model, optimizer, cfg)


if __name__ == "__main__":
    main()
