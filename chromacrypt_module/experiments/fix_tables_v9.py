import re

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# First clean up any accidental mess
text = text.replace('tex += "\\begin{adjustbox}{max width=\\linewidth}\\n"\n', '')
text = text.replace('\\n\\end{adjustbox}', '')

# The exact string in the python file is:
# tex += "\\begin{tabular}
# Which means text.find('tex += "\\\\begin{tabular}')
text = text.replace('tex += "\\\\begin{tabular}', 'tex += "\\\\begin{adjustbox}{max width=\\\\linewidth}\\n"\\n    tex += "\\\\begin{tabular}')
text = text.replace('tex += "\\\\end{tabular}', 'tex += "\\\\end{tabular}\\n\\\\end{adjustbox}')

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Finally injected!")
