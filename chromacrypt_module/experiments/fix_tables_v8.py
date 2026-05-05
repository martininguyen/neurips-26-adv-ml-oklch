import re

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Make sure we don't double wrap by stripping any existing adjustbox
text = text.replace('\\\\begin{adjustbox}{max width=\\\\linewidth}\\n', '')
text = text.replace('\\\\end{adjustbox}\\n', '')

# We will look for: tex += "\\begin{tabular}...
# And replace it with: tex += "\\begin{adjustbox}{max width=\linewidth}\n\\begin{tabular}...
# Wait, inside a Python raw string? No, text is the literal file contents.
# So it's: tex += "\\begin{tabular}
text = re.sub(r'(tex \+= "\\\\begin\{tabular\})', r'tex += "\\\\begin{adjustbox}{max width=\\\\linewidth}\\n"\n    \1', text)

# For the end, we look for: tex += "\\end{tabular}
# And replace it with: tex += "\\end{tabular}\n\\end{adjustbox}
text = re.sub(r'(tex \+= "\\\\end\{tabular\})', r'tex += "\\\\end{tabular}\\n\\\\end{adjustbox}', text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Injected adjustbox into generate_latex_tables.py")
