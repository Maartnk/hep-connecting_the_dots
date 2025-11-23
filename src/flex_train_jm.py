import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np
import os
import sys
import toml
import logging
import wandb
import argparse
from flex_model_jm import JointModel, TransformerRegressor, TransformerClassifier, generate_flex_padding_mask, generate_cluster_padding_mask, generate_sliding_window_padding_mask, token_pad_mask_from_seq_lengths, load_encreg
import utils.metrics_calculator as metrics_calculator
import utils.training_utils as training_utils
import utils.flex_data_utils as data_utils
import utils.output_utils as output_utils
import utils.wandb_utils as wandb_utils
from torch.amp import autocast, GradScaler

import math
import matplotlib.pyplot as plt
import pandas as pd

torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.float32)
    
def compute_losses(reg_criterion, cla_criterion, logits, labels, params_pred, params_true, pad_mask, alpha=1, beta=0.3):
    """Compute joint regression + classification losses.

    Parameters
    ----------
    reg_criterion:
        Regression loss module.
    cla_criterion:
        Classification loss module.
    logits:
        Tensor of shape (B, L, C)
    labels:
        Tensor of shape (B, L)
    params_pred:
        Predicted regression parameters of shape (B, L, D_pred).
    params_true:
        Ground-truth regression parameters of shape
        (B, L, D_true) or (B, L).
    pad_mask:
        Boolean tensor of shape (B, L) where True indicates padded tokens.
    alpha:
        Weight for regression loss.
    beta:
        Weight for classification loss.

    Returns
    -------
    tot_loss:
        Weighted sum: alpha * reg_loss + beta * cla_loss.
    reg_loss:
        Mean regression loss over valid tokens.
    cla_loss:
        Mean classification loss over non-padding tokens.
    """
    if params_true.dim() == 2:
        params_true = params_true.unsqueeze(-1)
    if params_pred.size(-1) != params_true.size(-1):
        D = min(params_pred.size(-1), params_true.size(-1))
        params_pred = params_pred[..., :D]
        params_true = params_true[..., :D]

    valid_reg = (~pad_mask) & (labels != 0)
    if valid_reg.any():
        reg_loss = reg_criterion(params_pred[valid_reg], params_true[valid_reg]).mean()
    else:
        reg_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    ce_per_token = cla_criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    cla_loss = ce_per_token[~pad_mask.view(-1)].mean()

    tot_loss = alpha * reg_loss + beta * cla_loss
    return tot_loss, reg_loss, cla_loss

def load_config(config_path):
    """
    Load the TOML configuration file and return a dictionary.
    """
    with open(config_path, "r") as config_file:
        config = toml.load(config_file)
    return config


def setup_logging(config, output_dir):
    """Configure root logger to log to both file and stdout."""
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def initialize_wandb(config, output_dir):
    wandb_logger = wandb_utils.WandbLogger(
        config=config["wandb"], output_dir=output_dir, job_type="training"
    )
    wandb_logger.initialize()
    return wandb_logger


def setup_training(config, device):
    """Construct model, optimizer, scheduler, and losses; optionally load checkpoint.

    Parameters
    ----------
    config:
        Full configuration dictionary loaded from TOML.
    device:
        Target device.

    Returns
    -------
    model:
        Joint regression + classification model on the given device.
    optimizer:
        AdamW optimizer over model parameters.
    lr_scheduler:
        ReduceLROnPlateau scheduler configured from config["training"]["scheduler"].
    reg_criterion:
        Regression loss (MSELoss with reduction="none").
    cla_criterion:
        Classification loss.
    start_epoch:
        Epoch index to start from, either 0 or checkpoint_epoch + 1.
    """
    model = JointModel(
        TransformerRegressor(
            inputfeature_dim=config["regressor_model"]["in_size"],
            num_params=config["regressor_model"]["out_size"],
            latent_dim=config["regressor_model"]["latent_dim"],
            num_heads=config["regressor_model"]["nr_heads"],
            embed_dim=config["regressor_model"]["embedding_size"],
            num_layers=config["regressor_model"]["num_encoder_layers"],
            dim_feedforward = config["regressor_model"]["hidden_dim"],
            dropout=config["regressor_model"]["dropout"],
            use_flash_attention=config["regressor_model"]["use_flashattn"],
        ),
        TransformerClassifier(
            inputfeature_dim=config["model"]["inputfeature_dim"],
            num_classes=config["data"]["num_classes"],
            num_heads=config["model"]["num_heads"],
            embed_dim=config["model"]["embed_dim"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward = config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            use_flash_attention=config["model"]["use_flash_attention"],
        )        
    ).to(device)

    # optimizer
    initial_lr = config["training"]["scheduler"]["initial_lr"]
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

    # scheduler
    mode = config["training"]["scheduler"]["mode"]
    factor = config["training"]["scheduler"]["factor"]
    patience = config["training"]["scheduler"]["patience"]
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience
    )

    # criterions
    reg_criterion = nn.MSELoss(reduction="none")
    cla_criterion  = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    # check whether to load from checkpoint
    if not config["training"]["start_from_scratch"]:
        if (
            "checkpoint_path" not in config["training"]
            or not config["training"]["checkpoint_path"]
        ):
            logging.error(
                "Checkpoint path must be provided when resuming from a checkpoint."
            )
            sys.exit(
                "Error: Checkpoint path not provided but required for resuming training."
            )
        elif not os.path.exists(config["training"]["checkpoint_path"]):
            logging.error(
                f"Checkpoint file not found: {config['training']['checkpoint_path']}"
            )
            sys.exit("Error: Checkpoint file does not exist.")
        else:
            checkpoint = torch.load(config["training"]["checkpoint_path"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            logging.info("Resuming training from checkpoint.")
    else:
        start_epoch = 0
        if (
            "checkpoint_path" in config["training"]
            and config["training"]["checkpoint_path"]
        ):
            logging.warning(
                "Checkpoint path provided but will not be used since training starts from scratch."
            )

    return model, optimizer, lr_scheduler, reg_criterion, cla_criterion, start_epoch


def train_epoch(
    model,
    trainloader,
    optimizer,
    scaler,
    reg_criterion,
    cla_criterion,
    device,
    config,
    epoch,
    metrics_calculator,
    wandb_logger,
    output_dir,
    ):
    """Run a single training epoch over the train loader."""
    model.train()  # Set model to training mode

    running_tot, running_loss_cla, running_loss_reg = 0., 0., 0.

    for i, (coords, params, labels, seq_lengths) in enumerate(trainloader):
        optimizer.zero_grad(set_to_none=True)

        coords, params, labels, seq_lengths = coords.to(device), params.to(device), labels.to(device), seq_lengths.to(device)
        flex_padding_mask = generate_flex_padding_mask(seq_lengths)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, params_pred, pad_mask = model(coords, f'train_{i}', flex_padding_mask, seq_lengths)

            tot_loss, mse_loss, ce_loss = compute_losses(reg_criterion, cla_criterion, logits, labels, params_pred, params, pad_mask)

        scaler.scale(tot_loss).backward()

        if config["logging"]["level"] == "DEBUG":
            wandb_logger.log_gradient_norm(model)

        scaler.step(optimizer)
        scaler.update()

        metrics_calculator.update(
            logits.view(-1, model.num_classes),
            labels.view(-1),
            loss=ce_loss.item()
        )

        running_tot += tot_loss.item()
        running_loss_cla  += ce_loss.item()
        running_loss_reg += mse_loss.item()        

    n_batches   = len(trainloader)
    epoch_loss  = running_tot / n_batches
    epoch_ce    = running_loss_cla  / n_batches
    epoch_mse   = running_loss_reg / n_batches
    epoch_accuracy   = metrics_calculator.calculate_accuracy()

    if epoch % config["logging"]["epoch_log_interval"] == 0:
        logging.info(f"Epoch {epoch+1} | "
                     f"train loss: {epoch_loss:.4f} "
                     f"(CE {epoch_ce:.4f} | MSE {epoch_mse:.4f})")
        logging.info(f"train accuracy: {epoch_accuracy:.2f}%")

    wandb_logger.log(
        {
            "train_loss": epoch_loss,
            "train_ce_loss": epoch_ce,
            "train_mse_loss": epoch_mse,
            "train_accuracy": epoch_accuracy,
            "epoch": epoch,
        }
    )

    if epoch % 10 == 0:
        epoch_score = metrics_calculator.calculate_trackml_score()
        logging.info(f"Training TrackML score: {epoch_score:.2f}%")
        wandb_logger.log({"train_score": epoch_score, "epoch": epoch})

    if epoch == 0:
        training_utils.log_memory_usage()


def validate_epoch(
    model,
    valloader,
    reg_criterion,
    cla_criterion,
    device,
    config,
    epoch,
    metrics_calculator,
    wandb_logger
    ):
    """Run a single validation epoch and return the mean validation loss."""
    model.eval()  # Set model to evaluation mode

    running_tot, running_loss_cla, running_loss_reg = 0., 0., 0.

    with torch.no_grad():
      for i, (coords, params, labels, seq_lengths) in enumerate(valloader):
        coords, params, labels, seq_lengths = coords.to(device), params.to(device), labels.to(device), seq_lengths.to(device)

        flex_padding_mask = generate_flex_padding_mask(seq_lengths)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, params_pred, pad_mask = model(coords, f'validation_{i}', flex_padding_mask, seq_lengths)

            tot_loss, reg_loss, cla_loss = compute_losses(reg_criterion, cla_criterion, logits, labels, params_pred, params, pad_mask)

        metrics_calculator.update(
            logits.view(-1, model.num_classes),
            labels.view(-1),
            loss=cla_loss.item()
        )
        
        running_tot += tot_loss.item()
        running_loss_cla += cla_loss.item()
        running_loss_reg += reg_loss.item()            

    n_batches = len(valloader)
    epoch_loss = running_tot / n_batches
    epoch_cla = running_loss_cla  / n_batches
    epoch_reg = running_loss_reg / n_batches
    epoch_accuracy = metrics_calculator.calculate_accuracy()

    if epoch % config["logging"]["epoch_log_interval"] == 0:
        logging.info(f"Epoch {epoch+1} | "
                     f"val loss: {epoch_loss:.4f} "
                     f"(CE {epoch_cla:.4f} | MSE {epoch_reg:.4f})")
        logging.info(f"val accuracy: {epoch_accuracy:.2f}%")

    wandb_logger.log(
        {
            "val_loss": epoch_loss,
            "val_ce_loss": epoch_cla,
            "val_mse_loss": epoch_reg,
            "val_accuracy": epoch_accuracy,
            "epoch": epoch,
        }
    )

    if epoch % 10 == 0:
        epoch_score = metrics_calculator.calculate_trackml_score()
        logging.info(f"Val TrackML score: {epoch_score:.2f}%")
        wandb_logger.log({"val_score": epoch_score, "epoch": epoch})

    return epoch_loss


def test(
    model,
    testloader,
    helperloader,
    truths_df,
    device,
    wandb_logger
    ):
    """Evaluate the model on the test set and log accuracy and TrackML scores."""
    model.eval()
    test_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)
    
    with torch.no_grad():
        for i, ((coords, _, labels, seq_lengths), (hit_ids, event_ids, _)) in enumerate(zip(testloader, helperloader)):
            coords, labels, seq_lengths = coords.to(device), labels.to(device), seq_lengths.to(device)

            flex_padding_mask = generate_flex_padding_mask(seq_lengths)         
                 
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _, _ = model(coords, f'test_{i}', flex_padding_mask, seq_lengths)

            # Build token pad mask (True = padded)
            B, L = labels.size(0), coords.size(1)
            pad_mask = token_pad_mask_from_seq_lengths(seq_lengths, L)

            # Flatten predictions and labels
            outputs = logits.view(-1, model.num_classes)
            flat_logits = outputs.argmax(dim=-1)
            flat_labels = labels.view(-1)
            event_ids_flat = event_ids.view(-1)
            hit_ids_flat = hit_ids.view(-1)

            # Keep only non-background and non-padding tokens
            mask_real = (flat_labels != 0) & (~pad_mask.view(-1))
            flat_predicted_real = flat_logits[mask_real]
            event_ids_real = event_ids_flat[mask_real.cpu()]
            hit_ids_real = hit_ids_flat[mask_real.cpu()]

            test_metrics_calculator.update(outputs.detach().cpu(), flat_labels.detach().cpu())
            test_metrics_calculator.add_true_score(
                hit_ids_real, event_ids_real, flat_predicted_real, truths_df
            )

    accuracy = test_metrics_calculator.calculate_accuracy()
    score = test_metrics_calculator.calculate_trackml_score()
    all_true_scores = test_metrics_calculator.get_all_true_scores()
    true_score = np.mean(all_true_scores) if all_true_scores else 0

    logging.info(f"Test accuracy: {accuracy:.2f}%")
    logging.info(f"Test TrackML score: {score:.2f}%")
    logging.info(f"Test true score: {true_score:.2f}%")
    wandb_logger.log(
        {"test_accuracy": accuracy, "test_score": score, "true_score": true_score}
    )


def main(config_path):
    config = load_config(config_path)
    output_dir = output_utils.unique_output_dir(config)  # with time stamp
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    wandb_logger = initialize_wandb(config, output_dir)
    logging.info(f"output_dir: {output_dir}")
    early_stopper = training_utils.EarlyStopping(
        config["training"]["early_stopping"], output_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    scaler = GradScaler()

    model, optimizer, lr_scheduler, reg_criterion, cla_criterion, start_epoch = setup_training(
        config, device
    )
    
    loaders = data_utils.load_dataloader(config, device, mode="all")
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    helper_loader = loaders["test_helper"]
    train_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)
    val_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)

    logging.info("Started training and validation")
    training_utils.log_memory_usage()
    if "watch_interval" in config["wandb"]:
        watch_interval = config["wandb"]["watch_interval"]
        wandb_logger.run.watch(model, log_freq=watch_interval)
        logging.info(f"wandb started watching at interval {watch_interval} ")
    for epoch in range(start_epoch, config["training"]["total_epochs"]):
        train_metrics_calculator.reset()
        val_metrics_calculator.reset()

        train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            reg_criterion,
            cla_criterion,
            device,
            config,
            epoch,
            train_metrics_calculator,
            wandb_logger,
            output_dir,
        )

        val_loss = validate_epoch(
            model,
            val_loader,
            reg_criterion,
            cla_criterion,
            device,
            config,
            epoch,
            val_metrics_calculator,
            wandb_logger,
        )
        # adjust learning rate based on validation loss
        lr_scheduler.step(val_loss)
        if config["training"]["scheduler"]["verbose"]:
            current_lr = optimizer.param_groups[0]["lr"]  # get last lr
            logging.info(f"lr: {current_lr}")
            wandb_logger.log({"lr": current_lr})

        # stop training and checkpoint the model if val loss stops improving
        early_stopper(val_loss)
        if early_stopper.should_stop():
            logging.info("Early stopping triggered. Saving checkpoint.")
            wandb_logger.save_model(
                model,
                f"model_earlystop_epoch_{epoch}.pth",
                optimizer,
                lr_scheduler,
                epoch,
                output_dir,
            )
            logging.info("Checkpoint saved to output_dir.")
            break
        # learning rate warm-up
        training_utils.adjust_learning_rate(optimizer, epoch, config)

        if epoch % config["logging"]["model_save_interval"] == 0:
            wandb_logger.save_model(
                model,
                f"model_epoch_{epoch}.pth",
                optimizer,
                lr_scheduler,
                epoch,
                output_dir,
            )

    logging.info("Finished training.")
    wandb_logger.save_model(
        model, "model_final.pth", optimizer, lr_scheduler, epoch, output_dir
    )
    logging.info("Checkpoint saved to output_dir.")
    truths_df = data_utils.load_truths(config)
    test(model, test_loader, helper_loader, truths_df, device, wandb_logger)
    logging.info("Finished testing")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given config file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the configuration TOML file."
    )

    args = parser.parse_args()
    main(args.config_path)
