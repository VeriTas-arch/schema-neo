"""
This script runs multiple Python scripts located in the `src/generator` directory
to automatically generate analysis results.
"""

import subprocess
import sys
from pathlib import Path

generator_dir = Path(__file__).resolve().parent.parent / "src/generator"

analysis_scripts = [
    "subspace_analysis.py",
    "subspace_analysis_1.py",
    "subspace_analysis_2.py",
    "subspace_analysis_3.py",
    "subspace_analysis_cross_layer_principal_angle.py",
    "subspace_analysis_for_different_cue.py",
    "subspace_analysis_for_different_cue_sub_cor.py",
    # "subspace_analysis_for_different_noise.py",
    "subspace_analysis_for_different_time_ang.py",
    "subspace_analysis_for_engvalue.py",
    "subspace_analysis_for_engvalue_for_different_cue.py",
    "subspace_analysis_for_rev.py",
    # "subspace_analysis_sen_mom.py",
    # "subspace_analysis_sen_mom_b.py",
]

for script in analysis_scripts:
    file_path = generator_dir / script
    if file_path.exists():
        print(f"Running {script} ...")
        python_exec = sys.executable
        result = subprocess.run(
            [python_exec, str(file_path)], capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            if "Error" in result.stderr or "Traceback" in result.stderr:
                print("Error:", result.stderr)
            else:
                print("Stderr:", result.stderr)
    else:
        print(f"{script} not found.")
