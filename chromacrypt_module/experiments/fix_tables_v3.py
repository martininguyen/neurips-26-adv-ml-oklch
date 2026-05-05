import os

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

wide_tables = [
    "build_table_channel_ablation",          # Table 1
    "build_table_autoattack_comparison",     # Table 2
    "build_table_full_benchmark",            # Table 3
    "build_table_diffusion_purification",    # Table 5
    "build_table_sd35_purification",         # Table 6
    "build_table_transferability_ablation",  # Table 8
    "build_table_structured_baselines"       # Table 10
]

for func in wide_tables:
    start = text.find(f"def {func}()")
    if start == -1: continue
    end = text.find("def build_", start + 10)
    if end == -1: end = len(text)
    
    body = text[start:end]
    # Change to table*
    body = body.replace('\\begin{table}[htbp]', '\\begin{table*}[htbp]')
    body = body.replace('\\begin{table}[h]', '\\begin{table*}[htbp]')
    body = body.replace('\\end{table}', '\\end{table*}')
    
    # Remove resizebox from Table 2 if it's there
    if func == "build_table_autoattack_comparison":
        body = body.replace('\\resizebox{\\columnwidth}{!}{%\\n\\begin{tabular}', '\\begin{tabular}')
        body = body.replace('\\end{tabular}\\n}\\n\\end{table*}', '\\end{tabular}\\n\\end{table*}')
        
    text = text[:start] + body + text[end:]

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Converted wide tables to table*")
