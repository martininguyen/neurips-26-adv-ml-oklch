import os

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replaces ALL tabular starts with adjustbox prefix
# The exact literal string is: tex += "\\begin{tabular}
text = text.replace('tex += "\\\\begin{tabular}', 'tex += "\\\\begin{adjustbox}{max width=\\\\linewidth}\\n"\\n    tex += "\\\\begin{tabular}')

# The exact literal string is: tex += "\\end{tabular}
text = text.replace('tex += "\\\\end{tabular}', 'tex += "\\\\end{tabular}\\n\\\\end{adjustbox}')

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Actually fixed!")
