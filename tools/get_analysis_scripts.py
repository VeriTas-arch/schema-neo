"""
This script automatically gets the analysis scripts from the `src/generator` directory
and returns a list of their names.
"""

from pathlib import Path

generator_dir = Path(__file__).resolve().parent.parent / "src/generator"

analysis_scripts = []

for script in generator_dir.glob("subspace_analysis*.py"):
    analysis_scripts.append(script.name)

# sort the list for better readability
analysis_scripts.sort()

print(analysis_scripts)
