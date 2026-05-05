import re

path = r"c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\generate_latex_tables.py"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

replacements = [
    (r'\\caption\{Continuous Perceptual Noise Ablation:.*?\}\\n', r'\\caption{Continuous Perceptual Noise Ablation ($N=1{,}000$). Tracking the Attack Success Rate (ASR) against ConvNeXt-L across a dynamically sweeping perceptual footprint (LPIPS) for both random RGB noise and the structural Chromic Grid.}\\n'),
    (r'\\caption\{White-Box Channel Ablation \(PGD\)\..*?\}\\n', r'\\caption{White-Box Channel Ablation (PGD). Efficacy mapped by isolating gradient steps to specific OKLCH axes.}\\n'),
    (r'\\caption\{Black-Box Transferability matrix targeting surrogate gradients from \{surrogate\}\..*?\}\\n', r'\\caption{Black-Box Transferability matrix targeting surrogate gradients from {surrogate}.}\\n'),
    (r'\\caption\{Cross-Architecture Vulnerability Diagnostic\..*?\}\\n', r'\\caption{Cross-Architecture Vulnerability Diagnostic. Evaluating the attack success rate of OKLCH optimization vectors across convolutional and transformer-based discriminators.}\\n'),
    (r'\\caption\{Structural Trap SOTA Diagnostic \(\$N=\{n_str\}\$\)\..*?\}\\n', r'\\caption{Structural Trap Benchmark Diagnostic ($N={n_str}$). Evaluating the attack success rate of the Chromic Grid structural harmonic against baseline and robust models.}\\n'),
    (r'\\caption\{Gradient Obfuscation Diagnostic \(\$N=1\{,\}000\$\)\..*?\}\\n', r'\\caption{Gradient Obfuscation Diagnostic ($N=1{,}000$). Evaluating identically restricted $L_{\\infty}$ gradient trajectories ($\\leq\\epsilon=4/255$) executed by the AutoAttack ensemble on core baseline and robust architectures.}\\n'),
    (r'\\caption\{Diffusion Purification Vulnerability \(N=795\).*?\}\\n', r'\\caption{Diffusion Purification Vulnerability ($N=795$) detailing the dose-response structural endurance (SNR) curve of the Chromic Grid against a ResNet-50 Target Classifier.}\\n'),
    (r'\\caption\{Structured Baseline Diagnostic \(\$N=1\{,\}000\$\)\..*?\}\\n', r'\\caption{Structured Baseline Diagnostic ($N=1{,}000$). Benchmarking the efficacy of the Chromic Grid\'s global spatial harmonic against localized Adversarial Patches across architectures.}\\n'),
    (r'\\caption\{Stable Diffusion 3\.5 Latent Projection Vulnerability \(\$N=795\$\)\..*?\}\\n', r'\\caption{Stable Diffusion 3.5 Latent Projection Vulnerability ($N=795$). Evaluating the purification efficacy of flow-matching architecture against structural geometries across varying amplitudes.}\\n'),
    (r'\\caption\{Classical and Generative Purification Failure\..*?\}\\n', r'\\caption{Classical and Generative Purification Evaluation. Benchmarking structurally explicit attacks against classical image processing filters and generative denoising pipelines.}\\n')
]

for pat, rep in replacements:
    text = re.sub(pat, rep, text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Captions updated in generate_latex_tables.py")
