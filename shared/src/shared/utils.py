import subprocess
from pathlib import Path


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
