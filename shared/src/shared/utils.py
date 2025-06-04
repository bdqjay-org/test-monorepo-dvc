import json
import subprocess
from pathlib import Path
from typing import Sequence
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_git_root() -> Path:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return Path(output.strip())
    except subprocess.CalledProcessError:
        raise RuntimeError("Not inside a Git repository")


def save_confusion_matrix(
    cm: Union[list, "np.ndarray"],
    labels: Sequence[str],
    save_dir: Path,
) -> None:
    """
    Save confusion matrix as JSON and plot heatmap PNG.

    Args:
        cm: Confusion matrix as 2D list or numpy array.
        labels: List of class label names.
        save_dir: Directory where files will be saved.

    Saves:
        - confusion_matrix.json (confusion matrix data)
        - confusion_matrix.png (confusion matrix heatmap)
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(cm)

    json_path = save_dir / "confusion_matrix.json"
    with open(json_path, "w") as f:
        json.dump({"confusion_matrix": cm_array.tolist()}, f)

    fig, ax = plt.subplots(
        figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.5))
    )
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, ha="right", rotation_mode="anchor")

    plot_path = save_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"Confusion matrix saved as JSON: {json_path}")
    print(f"Confusion matrix plot saved as PNG: {plot_path}")
