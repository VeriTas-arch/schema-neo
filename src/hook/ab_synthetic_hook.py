from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lib import utils

CUDA = torch.cuda.is_available()


class TrainingHook:
    def __init__(
        self,
        save_dir: str | Path = None,
        save_interval: int = 1,
        visualization_interval: int = 1,
        log_interval: int = 10,
        max_samples: int = 2,
        enable_checkpoint: bool = True,
        enable_visualization: bool = True,
        enable_logging: bool = True,
        logger: Logger = None,
    ):
        r"""
        A unified hook for logging, checkpointing and visualization.

        Parameters
        ----------
        save_interval: int
            epochs between saves
        visualization_interval: int
            epochs between visualizations
        log_interval: int
            batches between logs
        save_dir: Path|str
            directory to save outputs
        max_samples: int
            samples to visualize per epoch
        enable_logging: bool
            whether to enable logging
        enable_checkpoint: bool
            whether to enable checkpoint saving
        enable_visualization: bool
            whether to enable visualization saving
        logger: Logger
            logger for logging outputs
        """
        self.save_interval = save_interval
        self.visualization_interval = visualization_interval
        self.log_interval = log_interval
        self.save_dir = Path(save_dir) if save_dir else Path("./output")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.enable_logging = enable_logging
        self.enable_checkpoint = enable_checkpoint
        self.enable_visualization = enable_visualization
        self.logger = logger

        if self.enable_logging and self.logger is None:
            raise ValueError("Logger must be provided if logging is enabled.")

        # tracking
        self._train_batch_losses: list[float] = []
        self.train_epoch_losses: list[float] = []
        self.val_epoch_losses: list[float] = []

        # gradient norms per batch and epoch means
        self._grad_norms: list[float] = []
        self._epoch_grad_means: list[float] = []

        # dirs
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.visualization_dir = self.save_dir / "visualizations"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

    def before_epoch(self, epoch, cfg):
        if CUDA:
            torch.cuda.empty_cache()
        if self.enable_logging:
            self.logger.info(f"Starting epoch {epoch + 1} / {cfg.total_epochs}")

    def after_batch(self, batch_idx, grad_norm, output):
        """Called after each batch. Optionally compute grad norms if model given."""
        # delegate to helpers
        self._record_batch_metrics(output)
        if self.enable_logging and batch_idx % self.log_interval == 0:
            if grad_norm is not None:
                self._grad_norms.append(grad_norm)
                self.logger.info(
                    f"Batch {batch_idx}, Loss: {float(output["record_res"]["loss"]):.4f}, Original GradNorm: {grad_norm:.4f}"
                )
            else:
                self.logger.info(
                    f"Batch {batch_idx}, Loss: {float(output["record_res"]["loss"]):.4f}"
                )

    def after_epoch(self, epoch, model, optimizer, val_loader, cfg):
        """Called at the end of each epoch: validate, save checkpoint/visuals, aggregate metrics."""
        if CUDA:
            torch.cuda.synchronize()
        # validate and collect epoch metrics
        val_loss = utils.validate(model, val_loader, cfg)

        if self.enable_logging:
            self.logger.info(f"Val Loss: {val_loss:.4f}")

        self._record_epoch_metrics(val_loss)

        # checkpoint and visualization
        if self.enable_checkpoint and (epoch + 1) % self.save_interval == 0:
            self._save_checkpoint(epoch, model, optimizer, cfg)

        if self.enable_visualization and (epoch + 1) % self.visualization_interval == 0:
            self._save_visualizations(epoch, model, val_loader, cfg)

    def after_run(self, epoch, model, optimizer, cfg):
        """Called at the end of training: save latest checkpoint and summary plots."""
        # save latest
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.to_dict(),
            },
            Path(cfg.work_dir) / "latest.pth",
        )
        if self.enable_logging:
            self.logger.info(f"Training completed. Models are saved in {cfg.work_dir}")

        # plot and save aggregated curves
        self._plot_loss_curves()
        self._plot_grad_curve()

    #! ------------- RECORD FUNCTIONS -------------
    def _record_batch_metrics(self, output):
        """Record batch-level metrics (loss) and optionally grad norm."""
        batch_loss = float(output["record_res"]["loss"])
        self._train_batch_losses.append(batch_loss)

    def _record_epoch_metrics(self, val_loss):
        """Aggregate and reset per-epoch metrics."""
        if len(self._train_batch_losses) > 0:
            epoch_train_loss = float(np.mean(self._train_batch_losses))
        else:
            epoch_train_loss = float("nan")
        self.train_epoch_losses.append(epoch_train_loss)
        self.val_epoch_losses.append(float(val_loss))
        # reset batch losses for next epoch
        self._train_batch_losses = []

        # epoch-level grad norm mean
        if len(self._grad_norms) > 0:
            self._epoch_grad_means.append(float(np.mean(self._grad_norms)))
        else:
            self._epoch_grad_means.append(float("nan"))

    #! ------------- SAVE FUNCTIONS -------------
    def _save_checkpoint(self, epoch, model, optimizer, cfg):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.to_dict(),
            },
            self.checkpoint_dir / f"epoch_{epoch + 1}_model.pth",
        )

    def _save_visualizations(self, epoch, model, val_loader, cfg):
        data = next(iter(val_loader))
        output = utils.batch_processor(model, data, cfg)
        record_res = output["record_res"]

        for i in range(self.max_samples):
            utils.plot_model_performance(
                record_res,
                i,
                cfg.stim_dur,
                cfg.task,
                self.visualization_dir / f"epoch_{epoch + 1}_sample_{i + 1}.png",
            )

    #! ------------- PLOT FUNCTIONS -------------
    def _plot_loss_curves(self):
        plt.figure(figsize=(8, 5))
        plt.plot(
            np.arange(1, len(self.train_epoch_losses) + 1),
            self.train_epoch_losses,
            "-o",
            label="train_loss",
        )
        plt.plot(
            np.arange(1, len(self.val_epoch_losses) + 1),
            self.val_epoch_losses,
            "-o",
            label="val_loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        fig_dir = self.save_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(fig_dir)
        plt.close()
        if self.enable_logging:
            self.logger.info(f"Saved loss curve to {fig_dir}")

    def _plot_grad_curve(self):
        plt.figure(figsize=(8, 4))
        plt.plot(
            np.arange(1, len(self._grad_norms) + 1),
            self._grad_norms,
            "-",
            label="batch_grad_norm",
        )
        epoch_x = np.linspace(1, len(self._grad_norms), len(self._epoch_grad_means))
        plt.plot(epoch_x, self._epoch_grad_means, "o-", label="epoch_mean_grad")
        plt.xlabel("Batch index")
        plt.ylabel("Grad L2 norm")
        plt.title("Gradient norm per batch")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        fig_dir = self.save_dir / "grad_norms.png"
        plt.tight_layout()
        plt.savefig(fig_dir)
        plt.close()
        if self.enable_logging:
            self.logger.info(f"Saved grad norm curve to {fig_dir}")
