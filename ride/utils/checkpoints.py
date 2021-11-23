import os
from pathlib import Path


def latest_file_in(path: Path) -> Path:
    assert path.is_dir()
    paths = list(path.glob("*"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files found in {str(path)}")

    return Path(max(path.glob("*"), key=os.path.getctime))


def get_latest_checkpoint(log_dir: str) -> Path:
    logs_path = Path(log_dir)
    assert logs_path.is_dir()
    checkpoint_path = latest_file_in(logs_path / "checkpoints")
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoints found in {str(logs_path)}")
    assert checkpoint_path.exists()
    return checkpoint_path


def find_checkpoint(path: str) -> str:
    path = Path(path)
    assert path.exists()
    if "ckpt" in path.suffix:
        return str(path)

    latest_file = latest_file_in(path)
    if "ckpt" in latest_file.suffix:
        return str(latest_file)

    checkpoint_dir = path / "checkpoints"
    assert checkpoint_dir.exists()

    latest_file = latest_file_in(checkpoint_dir)
    if "ckpt" in latest_file.suffix:
        return str(latest_file)

    raise RuntimeError(f"Unable to find any checkpoint in {str(path)}")
