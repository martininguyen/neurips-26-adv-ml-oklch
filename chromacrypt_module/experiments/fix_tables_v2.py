import os
import re

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# For all tables EXCEPT Table 2 (AutoAttack) we want to REMOVE resizebox if it exists.
# We also want to make sure Table 8 (Transferability) has NO resizebox.
# But wait, Table 2 MUST HAVE resizebox so it shrinks to fit.

# Let's just do text replacements for exactly what's there.
# Pattern: tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}
# We want to replace it with: tex += "\\begin{tabular}

def remove_resizebox(func_name, code):
    start = code.find(f"def {func_name}()")
    if start == -1: return code
    end = code.find("def build_", start + 10)
    if end == -1: end = len(code)
    
    body = code[start:end]
    # Replace the starting tag
    body = body.replace('tex += "\\\\resizebox{\\\\columnwidth}{!}{%\\n\\\\begin{tabular}', 'tex += "\\\\begin{tabular}')
    body = body.replace('tex += "\\\\resizebox{\\\\columnwidth}{!}{\\n\\\\begin{tabular}', 'tex += "\\\\begin{tabular}')
    # Replace the ending tag
    body = body.replace('\\\\end{tabular}\\n}\\n\\\\end{table}', '\\\\end{tabular}\\n\\\\end{table}')
    
    return code[:start] + body + code[end:]

# Remove resizebox from all tables except Table 2
tables_to_unresize = [
    "build_table_random_noise",
    "build_table_channel_ablation", # Table 1
    "build_table_overhead",
    "build_table_transferability_ablation", # Table 8
    "build_table_cross_model",
    "build_table_full_benchmark",
    "build_table_diffusion_purification",
    "build_table_structured_baselines",
    "build_table_sd35_purification",
    "build_table_defenses"
]

for t in tables_to_unresize:
    text = remove_resizebox(t, text)

# Ensure Table 2 (AutoAttack) HAS resizebox
start = text.find("def build_table_autoattack_comparison()")
end = text.find("def build_table_diffusion_purification()")
body = text[start:end]
if "\\resizebox" not in body:
    body = body.replace('tex += "\\\\begin{tabular}', 'tex += "\\\\resizebox{\\\\columnwidth}{!}{%\\n\\\\begin{tabular}')
    body = body.replace('\\\\end{tabular}\\n\\\\end{table}', '\\\\end{tabular}\\n}\\n\\\\end{table}')
text = text[:start] + body + text[end:]

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Tables perfectly fixed!")
