import re
import os

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Remove \footnotesize
text = text.replace('\\footnotesize\\n', '')

# First, strip existing adjustbox if any (for idempotency)
text = text.replace('\\begin{adjustbox}{max width=\\linewidth}\\n', '')
text = text.replace('\\n\\end{adjustbox}', '')

# Now add them safely inside the python string
text = text.replace('\\begin{tabular}', '\\begin{adjustbox}{max width=\\linewidth}\\n\\begin{tabular}')
text = text.replace('\\end{tabular}', '\\end{tabular}\\n\\end{adjustbox}')

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Tables wrapped in adjustbox safely!")
