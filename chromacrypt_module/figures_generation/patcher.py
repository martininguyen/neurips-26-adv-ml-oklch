import os
import glob
import re

TARGET_DIR = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\figures_generation"
# Resolves to experiments/results/figures/<category_name_of_script>
RESULTS_PATH = r"os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), "

py_files = glob.glob(os.path.join(TARGET_DIR, "**", "*.py"), recursive=True)

for file in py_files:
    if os.path.basename(file) in ["patcher.py", "generate_all_legacy.py"]:
        continue
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ensure os is imported
    if "import os" not in content:
        content = "import os\n" + content
        
    def savefig_replacer(match):
        inner_content = match.group(1).strip()
        # Clean up any previously applied nested joins if we copied over corrupted ones
        if 'os.path.join' in inner_content:
            # Try to extract the literal string name dynamically if corrupted
            filename_match = re.search(r"['\"]([^'\"]+\.png)['\"]", inner_content)
            if filename_match:
                inner_content = "'" + filename_match.group(1) + "'"
                
        # inject the dynamic os.makedirs immediately preceding savefig securely to avoid crashes
        os_makedirs = f"os.makedirs({RESULTS_PATH[:-2]}), exist_ok=True)\n    "
        
        if inner_content.startswith("'") or inner_content.startswith('"'):
            replacement = RESULTS_PATH + inner_content + ")"
        else:
            replacement = RESULTS_PATH + inner_content + ")"
        return os_makedirs + f"plt.savefig({replacement},"

    new_content = re.sub(r'plt\.savefig\(\s*([^,)]+)\s*,', savefig_replacer, content)
    
    def savefig_replacer_nocomma(match):
        inner_content = match.group(1).strip()
        if 'os.path.join' in inner_content:
            filename_match = re.search(r"['\"]([^'\"]+\.png)['\"]", inner_content)
            if filename_match:
                inner_content = "'" + filename_match.group(1) + "'"
                
        os_makedirs = f"os.makedirs({RESULTS_PATH[:-2]}), exist_ok=True)\n    "
        return os_makedirs + f"plt.savefig({RESULTS_PATH}{inner_content}))"
        
    new_content = re.sub(r'plt\.savefig\(\s*([^,)]+)\s*\)', savefig_replacer_nocomma, new_content)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
print(f"Algorithm patching completed, cleanly sorting figures recursively via explicit bounds.")
