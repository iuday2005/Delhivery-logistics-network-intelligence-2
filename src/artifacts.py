"""Safe artifact loading helpers.

Large pickle files are kept out of GitHub. These helpers let the dashboard use
local artifacts when present and fall back to static summaries when deployed.
"""

from pathlib import Path
import pickle


def safe_load_pickle(artifact_dir: Path, file_name: str):
    """Return a pickle object, or None if the file is unavailable/unreadable."""
    file_path = artifact_dir / file_name
    try:
        with file_path.open("rb") as file:
            return pickle.load(file)
    except Exception:
        return None


def load_artifacts(artifact_dir: Path, artifact_files: dict) -> dict:
    """Load configured artifacts into a dictionary."""
    return {
        label: safe_load_pickle(artifact_dir, file_name)
        for label, file_name in artifact_files.items()
    }

