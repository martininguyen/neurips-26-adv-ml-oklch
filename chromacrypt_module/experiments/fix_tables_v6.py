import os

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix the broken literals
text = text.replace('tex += "\\begin{adjustbox}{max width=\linewidth}\n\\begin{tabular}', 'tex += "\\\\begin{adjustbox}{max width=\\\\linewidth}\\n\\\\begin{tabular}')
text = text.replace('\\end{tabular}\n\\end{adjustbox}\\n"', '\\\\end{tabular}\\n\\\\end{adjustbox}\\n"')

# And just in case any other broken literals exist:
# Let's completely undo and redo the adjustbox wrapping correctly.
text = text.replace('tex += "\\\\begin{adjustbox}{max width=\\\\linewidth}\\n\\\\begin{tabular}', 'tex += "\\\\begin{tabular}')
text = text.replace('tex += "\\\\end{tabular}\\n\\\\end{adjustbox}\\n"', 'tex += "\\\\end{tabular}\\n"')
text = text.replace('\\end{tabular}\\n\\end{adjustbox}\\n"', '\\end{tabular}\\n"')

# Now add it back properly:
text = text.replace('tex += "\\\\begin{tabular}', 'tex += "\\\\begin{adjustbox}{max width=\\\\linewidth}\\n\\\\begin{tabular}')
text = text.replace('\\\\end{tabular}\\n"', '\\\\end{tabular}\\n\\\\end{adjustbox}\\n"')

# Remove any remaining literal \end{adjustbox}
text = text.replace('\\end{tabular}\n\\end{adjustbox}', '\\end{tabular}')

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Tables fixed!")
