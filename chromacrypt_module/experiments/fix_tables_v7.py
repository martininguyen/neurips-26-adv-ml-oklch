import re
import os

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Remove any old resizebox or adjustbox to be safe
text = re.sub(r'\\resizebox\{\\columnwidth\}\{!\}\{%\n', '', text)
text = re.sub(r'\\resizebox\{\\columnwidth\}\{!\}\{\n', '', text)
text = text.replace('\\begin{adjustbox}{max width=\\linewidth}\n', '')
text = text.replace('\n\\end{adjustbox}', '')
text = re.sub(r'\\end\{tabular\}\n\}\n\\end', r'\\end{tabular}\n\\end', text)
text = re.sub(r'\\end\{tabular\*\}\n\}\n\\end', r'\\end{tabular*}\n\\end', text)

# 2. Change tabular* to tabular
text = text.replace('\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}', '\\begin{tabular}{@{}')
text = text.replace('\\end{tabular*}', '\\end{tabular}')

# 3. Add \small to all tables to help reduce width slightly
text = text.replace('\\centering\n\\small\n', '\\centering\n')
text = text.replace('\\centering\n', '\\centering\n\\small\n')

# 4. Use regex to wrap \begin{tabular} ... \end{tabular} with adjustbox
# Match \begin{tabular}{...} using regex: r'(\\begin\{tabular\}\{.*?\})'
text = re.sub(r'(\\begin\{tabular\}\{[^\}]+\})', r'\\begin{adjustbox}{max width=\\linewidth}\n\1', text)

# Match \end{tabular}
text = re.sub(r'(\\end\{tabular\})', r'\1\n\\end{adjustbox}', text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Tables flawlessly wrapped with adjustbox!")
