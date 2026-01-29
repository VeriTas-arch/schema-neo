"""
Scan model/<task>/<timestamp> folders and remove any timestamp folder that
does NOT contain a `latest.pth` file (search is recursive). This helps clean
partially interrupted runs that didn't produce a checkpoint.
"""

import shutil
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
model_dir = root_dir / "model"


def find_timestamp_folders(task_dir: Path):
    """Yield direct subdirectories of task_dir that look like timestamp folders.
    Timestamp format expected: YYYYMMDD_HHMMSS (e.g. 20251014_151841)
    """
    import re

    pattern = re.compile(r"^\d{8}_\d{6}$")
    if not task_dir.exists() or not task_dir.is_dir():
        return
    for child in task_dir.iterdir():
        if child.is_dir() and pattern.match(child.name):
            yield child


def has_latest_pth(folder: Path) -> bool:
    """Return True if folder contains a file named 'latest.pth' (recursively)."""
    try:
        for p in folder.rglob("latest.pth"):
            if p.is_file():
                return True
    except Exception:
        return False
    return False


def main():
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"Model dir does not exist or is not a directory: {model_dir}")
        sys.exit(1)

    tasks = [p for p in model_dir.iterdir() if p.is_dir()]
    if not tasks:
        print(f"No task subdirectories found under {model_dir}")
        return

    to_delete = []
    for task_dir in tasks:
        # Skip the temp folder if present
        if task_dir.name == "temp":
            continue
        for ts_folder in find_timestamp_folders(task_dir):
            if not has_latest_pth(ts_folder):
                to_delete.append(ts_folder)

    if not to_delete:
        print("No interrupted/empty timestamp folders found. Nothing to do.")
        return

    print("Found the following timestamp folders without latest.pth:")
    for p in to_delete:
        print(f"  {p}")

    # ask user interactively whether to proceed with deletion
    confirm = input("\nProceed to delete the above folders? Type 'y' to confirm: ")
    if confirm.strip().lower() not in ("y", "yes"):
        print("Aborting deletion (no folders were deleted).")
        return

    # perform deletions
    for p in to_delete:
        try:
            shutil.rmtree(p)
            print(f"Deleted: {p}")
        except Exception as e:
            print(f"Failed to delete {p}: {e}")


if __name__ == "__main__":
    main()
