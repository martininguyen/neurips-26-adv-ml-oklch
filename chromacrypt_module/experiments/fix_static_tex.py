import os
import re

results_dir = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\results"

for file in os.listdir(results_dir):
    if file.endswith(".tex"):
        fpath = os.path.join(results_dir, file)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        changed = False
        
        # Strip old resizebox just in case
        if "\\resizebox{\\columnwidth}{!}{" in content:
            content = content.replace('\\resizebox{\\columnwidth}{!}{', '')
            content = content.replace('}\n\\end{table}', '\n\\end{table}')
            content = content.replace('}\n\n\\end{table}', '\n\\end{table}')
            changed = True
        if "\\resizebox{\\columnwidth}{!}{%" in content:
            content = content.replace('\\resizebox{\\columnwidth}{!}{%', '')
            content = content.replace('}\n\\end{table}', '\n\\end{table}')
            changed = True
            
        # Ensure tabular is wrapped in adjustbox
        if "\\begin{adjustbox}" not in content and "\\begin{tabular}" in content:
            # We use regex to wrap the tabular block
            # Match \begin{tabular}{...} and wrap it
            content = re.sub(r'(\\begin\{tabular\}[^\n]*\n)', r'\\begin{adjustbox}{max width=\\linewidth}\n\1', content)
            content = content.replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}')
            changed = True
            
        if changed:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

print("Static files finally fixed!")
