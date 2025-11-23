import torch
import numpy as np
import os
import logging


def log_memory_usage():
    logging.info(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    logging.info(f"Memory Reserved:  {torch.cuda.memory_reserved() / 1e9} GB")


def adjust_learning_rate(optimizer, epoch, config):
    """Adjusts learning rate for each epoch during the warm-up phase, based on TOML config."""
    warmup_epochs = config["training"]["scheduler"]["warmup_epochs"]
    if epoch < warmup_epochs:
        initial_lr = config["training"]["scheduler"]["initial_lr"]
        target_lr = config["training"]["scheduler"]["target_lr"]
        current_lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, config, output_dir):
        self.patience = config["patience"]
        self.verbose = config["verbose"]
        self.path = os.path.join(output_dir, "earlystop_checkpoint.pt")
        self.counter = 0  # track num of epochs with no improvement
        self.early_stop = False
        self.best_val_loss = np.inf

    def __call__(self, val_loss):
        """
        Evaluates whether early stopping should be triggered based on the latest validation loss.
        - val_loss: The validation loss for the current epoch.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.verbose:
                logging.info(f"Validation loss decreased to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"No improvement in validation loss for {self.counter} consecutive epochs."
                )

            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(
                    f"Early stopping triggered. Best val_loss: {self.best_val_loss:.6f}, Last val_loss: {val_loss:.6f}"
                )

    def should_stop(self):
        return self.early_stop
