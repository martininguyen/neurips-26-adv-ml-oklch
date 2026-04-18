import os
import glob
import subprocess

TARGET_DIR = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\figures_generation"
VENV_PYTHON = r"c:\Users\marti\Documents\Projects\research_lab\working_scripts\venv_win\Scripts\python.exe"

py_files = glob.glob(os.path.join(TARGET_DIR, "**", "fig_*.py"), recursive=True)

for file in py_files:
    print(f"\nEvaluating visual execution geometry for -> {os.path.basename(file)}")
    try:
        # Run explicitly in the context of the script directory natively logically
        result = subprocess.run([VENV_PYTHON, file], cwd=os.path.dirname(file), capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Sub-Routine WARNING: {result.stderr}")
    except Exception as e:
        print(f"Error mapping pipeline boundaries natively: {e}")
