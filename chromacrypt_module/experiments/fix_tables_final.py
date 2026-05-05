import re
import os

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Replace tabular*
text = text.replace('\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}', '\\begin{tabular}{@{}')
text = text.replace('\\end{tabular*}', '\\end{tabular}')

# 2. Add \small
text = text.replace('\\centering\n', '\\centering\n\\small\n')

# 3. Handle \resizebox specifically by replacing the literal string
text = text.replace('tex += "\\resizebox{\\columnwidth}{!}{%\\n"\n', '')
text = text.replace('tex += "\\resizebox{\\columnwidth}{!}{\\n"\n', '')
# Also replace the closing brace of resizebox
text = text.replace('tex += "}\\n\\end{table}\\n"\n', 'tex += "\\end{table}\\n"\n')

# 4. Add adjustbox around tabular using exact literal replacements
text = text.replace('tex += "\\begin{tabular}', 'tex += "\\\\begin{adjustbox}{max width=\\\\linewidth}\\n\\\\begin{tabular}')
text = text.replace('tex += "\\end{tabular}\\n"', 'tex += "\\\\end{tabular}\\n\\\\end{adjustbox}\\n"')

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

# Now manually fix all .tex files in results/ that are static
results_dir = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\results"
for file in os.listdir(results_dir):
    if file.endswith(".tex"):
        fpath = os.path.join(results_dir, file)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        
        changed = False
        if "\\resizebox{\\columnwidth}{!}{" in content:
            content = content.replace('\\resizebox{\\columnwidth}{!}{', '\\begin{adjustbox}{max width=\\linewidth}')
            # We must replace the closing '}' with \end{adjustbox}
            # Look for the last '}' before \end{table}
            content = content.replace('}\n\\end{table}', '\\end{adjustbox}\n\\end{table}')
            changed = True
        elif "\\resizebox{\\columnwidth}{!}{%" in content:
            content = content.replace('\\resizebox{\\columnwidth}{!}{%', '\\begin{adjustbox}{max width=\\linewidth}')
            content = content.replace('}\n\\end{table}', '\\end{adjustbox}\n\\end{table}')
            changed = True
            
        if changed:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

print("Tables flawlessly fixed!")
