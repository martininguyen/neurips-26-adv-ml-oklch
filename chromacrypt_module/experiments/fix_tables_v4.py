import os
import re

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Remove all \resizebox{\columnwidth}{!}{% and \resizebox{\columnwidth}{!}{
text = re.sub(r'\\resizebox\{\\columnwidth\}\{!\}\{%\n', '', text)
text = re.sub(r'\\resizebox\{\\columnwidth\}\{!\}\{\n', '', text)
# Remove the closing brace of resizebox. This usually occurs right before \end{table} or \end{table*}
text = re.sub(r'\\end\{tabular\}\n\}\n\\end\{table\}', r'\\end{tabular}\n\\end{table}', text)
text = re.sub(r'\\end\{tabular\}\n\}\n\\end\{table\*\}', r'\\end{tabular}\n\\end{table*}', text)
text = re.sub(r'\\end\{tabular\*\}\n\}\n\\end\{table\}', r'\\end{tabular*}\n\\end{table}', text)

# 2. Revert table* to table
text = text.replace('\\begin{table*}', '\\begin{table}')
text = text.replace('\\end{table*}', '\\end{table}')

# 3. Revert tabular* to tabular
text = text.replace('\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}', '\\begin{tabular}{@{}')
text = text.replace('\\end{tabular*}', '\\end{tabular}')

# 4. Standardize font size to \footnotesize
# First, remove any existing \small or \footnotesize
text = text.replace('\\centering\n\\small\n', '\\centering\n')
text = text.replace('\\centering\n\\footnotesize\n', '\\centering\n')
# Now add \footnotesize to all
text = text.replace('\\centering\n', '\\centering\n\\footnotesize\n')

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Standardized all tables to \footnotesize with no resizebox")
