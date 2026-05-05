import os
import glob
import subprocess

TARGET_DIR = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\figures_generation"
VENV_PYTHON = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\venv_win\Scripts\python.exe"

# Explicitly bounded script execution pipeline to prevent wildcard evaluation of dormant or broken internal test files
py_files = [
    os.path.join(TARGET_DIR, "main_benchmark_chart", "fig_main_benchmark_chart.py"),
    os.path.join(TARGET_DIR, "nyquist_evasion_proof", "fig_nyquist_evasion.py"),
    os.path.join(TARGET_DIR, "robust_model_failures", "fig_robust_model_failures.py"),
    os.path.join(TARGET_DIR, "visual_proof_grids", "fig_visual_proof_grids.py"),
    os.path.join(TARGET_DIR, "vit_vs_cnn_frequency", "fig_vit_vs_cnn_frequency.py"),
    os.path.join(TARGET_DIR, "vit_vs_cnn_frequency", "fig_vit_patch_fft.py")
]

env = os.environ.copy()
env["PYTHONPATH"] = r"c:\Users\marti\Documents\Projects\research_lab"

for file in py_files:
    print(f"\nEvaluating visual execution geometry for -> {os.path.basename(file)}")
    try:
        # Run explicitly in the context of the module root logically
        result = subprocess.run([VENV_PYTHON, file], cwd=r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module", env=env, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Sub-Routine WARNING: {result.stderr}")
    except Exception as e:
        print(f"Error mapping pipeline boundaries natively: {e}")
