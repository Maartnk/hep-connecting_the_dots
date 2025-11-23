
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["plot_heatmap_arrays", "plot_heatmaps_for_params"]

# Default parameter labels and indices (your pipeline)
DEFAULT_LABELS: List[str] = ["theta", "sin phi", "cos phi", "q"]
DEFAULT_INDEX_MAP: Dict[str, int] = {
    # Preferred names
    "theta": 0, "sin_phi": 1, "cos_phi": 2, "q": 3
}

def _coerce_arrays(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure inputs are 2D float arrays with same shape (N, 4)."""
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if y_pred.ndim == 3:
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])

    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred and y_true must have the same shape, got {y_pred.shape} vs {y_true.shape}")
    if y_pred.shape[1] != 4:
        raise ValueError(f"Expected 4 parameter columns, got {y_pred.shape[1]} columns")

    # Remove rows with NaN/Inf in either array
    ok = np.isfinite(y_pred).all(axis=1) & np.isfinite(y_true).all(axis=1)
    if ok.sum() == 0:
        raise ValueError("All rows contain NaN/Inf; nothing to plot.")
    if ok.sum() != ok.shape[0]:
        y_pred = y_pred[ok]
        y_true = y_true[ok]
    return y_pred, y_true


def plot_heatmap_arrays(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    param_idx: int,
    label: str,
    out_path: str,
    bins: int = 100,
) -> None:
    """
    Create a Predicted-vs-True 2D histogram heatmap for a single parameter.

    Parameters
    ----------
    y_pred : (N, 4) array
        Predicted parameters (flattened over all non-padded hits).
    y_true : (N, 4) array
        True parameters aligned with y_pred.
    param_idx : int
        Column index in [0..3] selecting the parameter to plot.
    label : str
        User-facing label, e.g., "theta", "sinphi", "cosphi", "q".
    out_path : str
        Path to write a PNG file.
    bins : int
        Number of bins for both x and y (square histogram).
    """
    y_pred, y_true = _coerce_arrays(y_pred, y_true)

    if not (0 <= param_idx < 4):
        raise ValueError(f"param_idx must be in [0, 3], got {param_idx}")

    pred_col = y_pred[:, param_idx]
    true_col = y_true[:, param_idx]

    # 2D histogram of Predicted vs True
    heatmap, xedges, yedges = np.histogram2d(pred_col, true_col, bins=bins)

    # Mask zeros as NaN to make them white
    heatmap = np.where(heatmap == 0, np.nan, heatmap)

    # Prepare figure
    plt.figure(figsize=(8, 6), dpi=150)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    img = plt.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    cbar = plt.colorbar(img)
    cbar.set_label("Count", rotation=90)

    # Axis labels & title
    label_clean = label.replace("_", " ")
    plt.xlabel(f"Predicted {label_clean}")
    plt.ylabel(f"True {label_clean}")
    plt.title(f"Regressed {label_clean} vs ground truth")

    if label.lower() in {"q"}:
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)

    out_path = str(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmaps_for_params(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    output_dir: str,
    tag: str = "",
    bins: int = 100,
    labels: Optional[List[str]] = None,
    index_map: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """
    Plot heatmaps for a set of parameters, returning a mapping from parameter label to saved path.
    """
    if labels is None:
        labels = DEFAULT_LABELS
    if index_map is None:
        index_map = DEFAULT_INDEX_MAP

    # Normalize tag
    file_tag = f"_{tag}" if tag else ""

    # Coerce & basic validation once
    y_pred, y_true = _coerce_arrays(y_pred, y_true)

    saved: Dict[str, str] = {}
    for label in labels:
        key = label.lower()
        if key not in index_map:
            alt = key
            if key == "sin phi": alt = "sin_phi"
            elif key == "cos phi": alt = "cos_phi"
            elif key == "theta": alt = "theta"
            key = alt

        if key not in index_map:
            raise KeyError(f"Unknown parameter label '{label}'. Update index_map accordingly.")

        idx = index_map[key]
        out_path = os.path.join(output_dir, f"{label}_heatmap{file_tag}.png")
        plot_heatmap_arrays(y_pred, y_true, idx, label, out_path, bins=bins)
        saved[label] = out_path

    return saved
