import re

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace all \resizebox{\columnwidth}{!}{ \begin{tabular}...} with \begin{tabular}...
# Except for the ones with 8 columns: table_channel_ablation and table_transferability_ablation.
# Let's do it specifically for the functions we want.

def remove_resizebox(func_name, text):
    # Find the function body
    start = text.find(f"def {func_name}()")
    if start == -1: return text
    end = text.find("def build_", start + 10)
    if end == -1: end = text.find("if __name__ ==")
    
    body = text[start:end]
    # Replace resizebox begin
    body = re.sub(r'\\resizebox\{\\columnwidth\}\{!\}\{\\n\\begin\{tabular\}', r'\\begin{tabular}', body)
    # Replace resizebox end
    body = body.replace(r'\end{tabular}\n}\n\end{table}', r'\end{tabular}\n\end{table}')
    
    return text[:start] + body + text[end:]

for f_name in [
    "build_table_overhead",
    "build_table_cross_model",
    "build_table_full_benchmark",
    "build_table_diffusion_purification",
    "build_table_structured_baselines",
    "build_table_sd35_purification",
    "build_table_defenses"
]:
    text = remove_resizebox(f_name, text)

# Also fix autoattack comparison which uses tabular*
start = text.find("def build_table_autoattack_comparison()")
end = text.find("def build_table_diffusion_purification()", start)
body = text[start:end]
body = body.replace(r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}', r'\begin{tabular}{@{}lccc@{}}')
body = body.replace(r'\end{tabular*}', r'\end{tabular}')
text = text[:start] + body + text[end:]

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Tables standardized in generate_latex_tables.py")
