import torch
import numpy as np
import os
import toml
import logging
import wandb
import argparse
from flex_model_jm import JointModel, TransformerRegressor, TransformerClassifier, generate_flex_padding_mask, token_pad_mask_from_seq_lengths
import utils.metrics_calculator as metrics_calculator
import utils.training_utils as training_utils
import utils.flex_data_utils as data_utils
import utils.output_utils as output_utils
import utils.wandb_utils as wandb_utils
import time

from torch.amp import autocast

from plotting_heatmaps import plot_heatmaps_for_params


def load_config(config_path):
    """
    Load the TOML configuration file and return a dictionary.
    """
    with open(config_path, "r") as config_file:
        config = toml.load(config_file)
    return config


def load_model(config, device):
    """
    Instantiate the JointModel and load weights from a checkpoint.

    Args:
        config: Parsed configuration mapping.
        device: Torch device to load the model onto.

    Returns:
        A JointModel instance in eval mode, loaded with checkpoint weights.

    Raises:
        SystemExit: If no checkpoint path is provided or the file does not exist.
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

    if (
        "checkpoint_path" not in config["model"]
        or not config["model"]["checkpoint_path"]
    ):
        logging.error("Checkpoint path must be provided for evaluation.")
    else:
        checkpoint = torch.load(config["model"]["checkpoint_path"])
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint["epoch"] + 1
        logging.info(
            f"Loaded model_state of epoch {epoch}. Ignoring optimizer_state and scheduler_state. Starting evaluation from checkpoint."
        )

    model.eval()
    return model


def setup_logging(config, output_dir):
    """
    Configure logging to both file and stdout.

    Args:
        config: Parsed configuration mapping.
        output_dir: Directory where log files will be written.
    """
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, "evaluation.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def initialize_wandb(config, output_dir):
    """
    Initialize a WandbLogger in 'evaluation' job mode.

    Args:
        config: Parsed configuration mapping.
        output_dir: Directory where wandb artifacts / models will be stored.

    Returns:
        An initialized WandbLogger instance.
    """
    wandb_logger = wandb_utils.WandbLogger(
        config=config["wandb"], output_dir=output_dir, job_type="evaluation"
    )
    wandb_logger.initialize()
    return wandb_logger



def evaluate(model, testloader, helperloader, truths_df, device, config, wandb_logger, output_dir, save_heatmaps=False, heatmap_tag=""):
    """
    Run inference on the test set, compute scoring metrics, and optionally
    generate regression heatmaps for predicted vs true parameters.

    Args:
        model: Trained JointModel in eval mode.
        testloader: DataLoader providing (coords, params_true, labels, seq_lengths).
        helperloader: DataLoader providing (hit_ids, event_ids, ...).
        truths_df: Ground-truth dataframe used by MetricsCalculator.add_true_score.
        device: Torch device for inference.
        wandb_logger: Experiment logger used for timing and score metrics.
        output_dir: Directory where heatmap images will be written.
        save_heatmaps: If True, aggregate all non-padded hits and save heatmaps.
        heatmap_tag: Tag passed through to the plotting function for filenames.
    """
    test_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)

    cpu_time_ms = []
    gpu_time_ms = []
    start_gpu_event = torch.cuda.Event(enable_timing=True)
    end_gpu_event = torch.cuda.Event(enable_timing=True)

    # Holders for heatmap data (collected across batches)
    pred_chunks = []
    true_chunks = []
    event_chunks = []


    with torch.no_grad():
        for i, ((coords, params_true, labels, seq_lengths), (hit_ids, event_ids, _)) in enumerate(zip(testloader, helperloader)):
            start_cpu_time = time.process_time_ns()
            coords = coords.to(device)
            params_true = params_true.to(device)
            labels = labels.to(device)
            seq_lengths = seq_lengths.to(device)
            flex_padding_mask = generate_flex_padding_mask(seq_lengths)
            end_cpu_time = time.process_time_ns()
            batch_cpu_time_ms = (end_cpu_time - start_cpu_time) / 1e6
            cpu_time_ms.append(batch_cpu_time_ms)

            start_gpu_event.record()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, params_pred, _ = model(coords, f'test_{i}', flex_padding_mask, seq_lengths)

            # Build token pad mask (True = padded)
            B, L = labels.size(0), coords.size(1)
            pad_mask = token_pad_mask_from_seq_lengths(seq_lengths, L)          # (B, L) bool

            # Flatten predictions and labels
            outputs = logits.view(-1, model.num_classes)
            flat_logits = outputs.argmax(dim=-1)
            flat_labels = labels.view(-1)
            event_ids_flat = event_ids.view(-1)
            hit_ids_flat = hit_ids.view(-1)

            # Keep only non-background and non-padding tokens for the true score
            mask_real = (flat_labels != 0) & (~pad_mask.view(-1))
            flat_predicted_real = flat_logits[mask_real]
            event_ids_real = event_ids_flat[mask_real.cpu()]
            hit_ids_real = hit_ids_flat[mask_real.cpu()]

            end_gpu_event.record()
            torch.cuda.synchronize()
            batch_gpu_time_ms = start_gpu_event.elapsed_time(end_gpu_event)
            gpu_time_ms.append(batch_gpu_time_ms)
            logging.debug(f"Batch cpu time: {batch_cpu_time_ms:.6f} ms")
            logging.debug(f"Batch gpu time: {batch_gpu_time_ms:.6f} ms")

            test_metrics_calculator.add_true_score(
                hit_ids_real, event_ids_real, flat_predicted_real, truths_df
            )
            nonpad_mask = (~pad_mask).view(B, L)  # (B, L) True for real tokens

            # Ensure params_pred and params_true have shape (B, L, 4)
            if params_pred.shape[-1] != 4 or params_true.shape[-1] != 4:
                raise ValueError(f"Expected 4 regressed parameters, got {params_pred.shape} vs {params_true.shape}")

            # Flatten to (B*L, 4) and select only non-padded hits
            P = params_pred.shape[-1]
            pred_flat = params_pred.reshape(B*L, P)
            true_flat = params_true.reshape(B*L, P)
            nonpad_flat = nonpad_mask.reshape(B*L)

            pred_sel = pred_flat[nonpad_flat]
            true_sel = true_flat[nonpad_flat]
            pred_np = pred_sel.to(dtype=torch.float32).cpu().numpy()
            true_np = true_sel.to(dtype=torch.float32).cpu().numpy()

            event_flat = event_ids.view(-1)
            event_np = event_flat[nonpad_flat.cpu()].cpu().numpy()

            pred_chunks.append(pred_np)
            true_chunks.append(true_np)
            event_chunks.append(event_np)

    # Average times
    avg_cpu_time_ms = sum(cpu_time_ms[1:]) / len(cpu_time_ms[1:]) if len(cpu_time_ms) > 1 else (cpu_time_ms[0] if cpu_time_ms else 0.0)
    avg_gpu_time_ms = sum(gpu_time_ms[1:]) / len(gpu_time_ms[1:]) if len(gpu_time_ms) > 1 else (gpu_time_ms[0] if gpu_time_ms else 0.0)
    logging.info(f"Avg CPU time: {avg_cpu_time_ms:.2f} ms")
    logging.info(f"Avg GPU time: {avg_gpu_time_ms:.2f} ms")
    wandb_logger.log(
        {
            "avg_cpu_time_ms": avg_cpu_time_ms,
            "avg_gpu_time_ms": avg_gpu_time_ms,
        }
    )

    all_true_scores = test_metrics_calculator.get_all_true_scores()
    true_score = np.mean(all_true_scores) if all_true_scores else 0

    logging.info(f"True score: {true_score:.2f}%")
    wandb_logger.log({"true_score": true_score})

    if save_heatmaps:
        if len(pred_chunks) == 0:
            logging.warning("No non-padded hits collected; skipping heatmap plotting.")
        else:
            pred_all = _np.concatenate(pred_chunks, axis=0)
            true_all = _np.concatenate(true_chunks, axis=0)
            saved = plot_heatmaps_for_params(pred_all, true_all, output_dir, tag=heatmap_tag, bins=100)
            for k, v in saved.items():
                logging.info(f"Saved {k} heatmap to: {v}")
            param_names = ["theta", "sinphi", "cosphi", "q"]
            print("\nResidual statistics (pred - true), across all non-padded hits:")
            for j, name in enumerate(param_names):
                err = (pred_all[:, j] - true_all[:, j]).astype(np.float64)
                n = err.size
                if n == 0:
                    print(f"  {name:7s} | N=0 -> no data")
                    continue
                bias = float(err.mean())
                std  = float(err.std(ddof=1)) if n > 1 else 0.0
                rmse = float(np.sqrt((err * err).mean()))
                mae  = float(np.abs(err).mean())
                logging.info(f"  {name:7s} | N={n:7d} | std={std:.6g} | bias={bias:.6g} | rmse={rmse:.6g} | mae={mae:.6g}")


def main(config_path):
    config = load_config(config_path)
    output_dir = output_utils.unique_output_dir(config)  # with time stamp
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    logging.info(f"output_dir: {output_dir}%")
    wandb_logger = initialize_wandb(config, output_dir)

    start_wallclock = time.perf_counter_ns()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = load_model(config, device)
    loaders = data_utils.load_dataloader(config, device, mode="eval")
    data_loader = loaders["test"]
    helper_loader = loaders["test_helper"]
    truths_df = data_utils.load_truths(config)

    logging.info("Started evaluation.")
    evaluate(model, data_loader, helper_loader, truths_df, device, config, wandb_logger, output_dir, save_heatmaps=True, heatmap_tag="TEST")
    logging.info("Finished evaluation.")
    end_wallclock = time.perf_counter_ns()
    total_wallclock_time_ms = (end_wallclock - start_wallclock) / 1e6
    logging.info(
        f"Total wallclock time (including scoring): {total_wallclock_time_ms:.2f} ms"
    )
    wandb_logger.log({"total_wallclock_time_ms": total_wallclock_time_ms})

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measuring the inference speed of a model with a given config file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the configuration TOML file."
    )

    args = parser.parse_args()
    main(args.config_path)
