import os

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # First, strip broken syntax
    if "tex += \"\\begin{adjustbox}" in line:
        continue
    if "tex += \"\\end{adjustbox}" in line:
        continue
    if "tex +=" in line and "\\end{tabular}\\n\\end{adjustbox}" in line:
        line = line.replace("\\end{tabular}\\n\\end{adjustbox}", "\\end{tabular}")
        
    if "tex += \"\\begin{tabular}" in line:
        # Prepend the adjustbox line exactly with proper indentation
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f'{indent}tex += "\\\\begin{{adjustbox}}{{max width=\\\\linewidth}}\\n"\n')
        new_lines.append(line)
    elif "tex += \"\\end{tabular}" in line:
        # We append the adjustbox end to the same line
        # line might be: tex += "\\end{tabular}\n"\n
        # Change it to add \end{adjustbox}
        line = line.replace('\\end{tabular}', '\\end{tabular}\\n\\\\end{adjustbox}')
        new_lines.append(line)
    else:
        new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Fixed line by line!")
