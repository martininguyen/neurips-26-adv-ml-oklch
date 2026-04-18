import os
import sys
import subprocess
import argparse
import glob

def run_suite(directory):
    scripts = sorted(glob.glob(os.path.join(directory, "benchmark_*.py")))
    if not scripts:
        print(f"No benchmark scripts natively found in {directory}.")
        return

    print("="*80)
    print(f"Executing {os.path.basename(directory).upper()} Architecture Validation Suite")
    print("="*80)

    for script in scripts:
        print(f"\\n---> Offloading independent memory trace for: {os.path.basename(script)}")
        # Isolating each script entirely clears the GPU stack variables dynamically natively
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"!!! FATAL ABORT: Iteration mapped in {os.path.basename(script)} decoupled with error code {result.returncode}. Skipping remaining nodes to preserve state... !!!")
            break

def main():
    parser = argparse.ArgumentParser(description="ChromaCrypt VRAM-Safe Orchestrator")
    parser.add_argument("--suite", choices=["discriminative", "generative", "all"], default="all",
                        help="Select execution domain. Evaluates independent mathematical geometries globally.")
    args = parser.parse_args()

    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    disc_dir = os.path.join(experiments_dir, "discriminative")
    gen_dir = os.path.join(experiments_dir, "generative")
    
    if args.suite in ["discriminative", "all"]:
        run_suite(disc_dir)
        
    if args.suite in ["generative", "all"]:
        run_suite(gen_dir)
        
    print("\n" + "="*80)
    print("Sequential Architecture Verification Terminally Concluded.")
    print("Initiating Latex Synthesis Pipeline...")
    
    latex_script = os.path.join(experiments_dir, "generate_latex_tables.py")
    if os.path.exists(latex_script):
        subprocess.run([sys.executable, latex_script], check=False)
    else:
         print("LaTeX compiler absent.")

if __name__ == "__main__":
    main()
