"""
This script automatically selects the latest `latest.pth` file for each task type
(forward/backward/switch) in the model directory, backs them up to model/temp,
and renames them accordingly.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
model_dir = root_dir / "model"
temp_dir = model_dir / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

task_types = ["forward", "backward", "switch"]
pattern_map = {t: re.compile(r"(\d{8}_\d{6})") for t in task_types}

for task in task_types:
    latest_time = None
    latest_folder = None
    # do not overwrite the top-level model_dir variable; use task_dir
    task_dir = model_dir / f"{task}"
    if not task_dir.exists() or not task_dir.is_dir():
        print(f"No folder found for task {task} (expected {task_dir})")
        continue

    for folder in task_dir.iterdir():
        if folder.is_dir():
            m = pattern_map[task].match(folder.name)
            if m:
                dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                if (latest_time is None) or (dt > latest_time):
                    latest_time = dt
                    latest_folder = folder

    if latest_folder:
        # prefer a top-level latest.pth inside the timestamped folder
        candidate = latest_folder / "latest.pth"
        latest_pth = None
        if candidate.exists():
            latest_pth = candidate
        else:
            # fallback: search recursively for any latest.pth file
            for pth in latest_folder.rglob("latest.pth"):
                latest_pth = pth
                break

        if latest_pth and latest_pth.exists():
            dst_path = temp_dir / f"{task}_latest.pth"
            shutil.copy2(latest_pth, dst_path)
            print(f"Copied {latest_pth} -> {dst_path}")
        else:
            print(f"No latest.pth found in {latest_folder}")
    else:
        print(f"No timestamped folder found for task {task} under {task_dir}")
