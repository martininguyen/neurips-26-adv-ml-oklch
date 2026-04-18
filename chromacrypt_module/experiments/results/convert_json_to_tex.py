import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def build_table5():
    with open(os.path.join(SCRIPT_DIR, "table5_channel_ablation.json"), "r") as f:
        data = json.load(f)
        
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{White-Box Channel Ablation (PGD). Efficacy mapped by isolating gradient steps to specific OKLCH axes. Freezing Lightness (L) drastically degrades attack success rates ($98.8\\% \\rightarrow 75.0\\%$) against geometric classifiers, confirming spatial luminosity as the primary vulnerability vector.}\n"
    tex += "\\label{tab:channel_ablation}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n"
    tex += "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
    tex += "\\textbf{Optimization Axes} & \\textbf{LPIPS} & \\textbf{ResNet50} & \\textbf{EfficientNet} & \\textbf{ViT-B-16} & \\textbf{VGG16} & \\textbf{Mean ASR} \\\\ \\midrule\n"
    for k, v in data.items():
        lpips = f"{v['lpips']:.4f}"
        rn = f"{v['asr'].get('ResNet50', 0):.1f}\\%"
        en = f"{v['asr'].get('EfficientNet', 0):.1f}\\%"
        vit = f"{v['asr'].get('ViT-B-16', 0):.1f}\\%"
        vgg = f"{v['asr'].get('VGG16', 0):.1f}\\%"
        mean = f"{v['mean_asr']:.1f}\\%"
        key_str = k.replace("$", "\\$") if "$" not in k else k # Keep math mode for key if exists
        tex += f"{key_str} & {lpips} & {rn} & {en} & {vit} & {vgg} & \\textbf{{{mean}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(SCRIPT_DIR, "table5_channel_ablation.tex"), "w") as f:
        f.write(tex)

def build_table6():
    with open(os.path.join(SCRIPT_DIR, "table6_whitebox_ablation.json"), "r") as f:
        data = json.load(f)
        
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{White-Box Chromatic Channel Ablation (Targeted Optimizations). Isolation mapping of geometric OKLCH dimensions verifies that L provides the overwhelming majority of adversarial viability. (Columns dictate target models).}\n"
    tex += "\\label{tab:whitebox_channel_ablation}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n"
    tex += "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n"
    tex += "\\textbf{Optimization Axes} & \\textbf{LPIPS} & \\textbf{ResNet50} & \\textbf{EfficientNet} & \\textbf{ViT-B-16} & \\textbf{VGG16} & \\textbf{Mean ASR} \\\\ \\midrule\n"
    for k, v in data.items():
        lpips = f"{v['lpips']:.4f}"
        rn = f"{v['asr'].get('ResNet50', 0):.1f}\\%"
        en = f"{v['asr'].get('EfficientNet', 0):.1f}\\%"
        vit = f"{v['asr'].get('ViT-B-16', 0):.1f}\\%"
        vgg = f"{v['asr'].get('VGG16', 0):.1f}\\%"
        mean = f"{v['mean_asr']:.1f}\\%"
        key_str = k.replace("$", "\\$") if "$" not in k else k
        tex += f"{key_str} & {lpips} & {rn} & {en} & {vit} & {vgg} & \\textbf{{{mean}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(SCRIPT_DIR, "table6_whitebox_ablation.tex"), "w") as f:
        f.write(tex)

def build_table7():
    with open(os.path.join(SCRIPT_DIR, "table7_transferability_ablation.json"), "r") as f:
        data = json.load(f)
        
    res = data.get("results", data)
    surrogate = data.get("metadata", {}).get("surrogate_model", "ResNet50")
        
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += f"\\caption{{Black-Box Transferability matrix targeting surrogate gradients from {surrogate}. Transfer success metrics identify that standard uncharacterized iterative perturbations fundamentally fail to generalize to Vision Transformers.}}\n"
    tex += "\\label{tab:transferability_matrix}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n"
    tex += "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
    tex += "\\textbf{Optimization Axes} & \\textbf{LPIPS} & \\textbf{AlexNet} & \\textbf{VGG16} & \\textbf{MobileNet} & \\textbf{EfficientNet} & \\textbf{ViT-B-16} & \\textbf{Mean ASR} \\\\ \\midrule\n"
    for k, v in res.items():
        lpips = f"{v['lpips']:.4f}"
        alx = f"{v['asr'].get('AlexNet', 0):.1f}\\%"
        vgg = f"{v['asr'].get('VGG16', 0):.1f}\\%"
        mob = f"{v['asr'].get('MobileNet', 0):.1f}\\%"
        eff = f"{v['asr'].get('EfficientNet', 0):.1f}\\%"
        vit = f"{v['asr'].get('ViT-B-16', 0):.1f}\\%"
        mean = f"{v['mean_asr']:.1f}\\%"
        tex += f"{k} & {lpips} & {alx} & {vgg} & {mob} & {eff} & {vit} & \\textbf{{{mean}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(SCRIPT_DIR, "table7_transferability_ablation.tex"), "w") as f:
        f.write(tex)

def build_table9():
    with open(os.path.join(SCRIPT_DIR, "table9_full_benchmark.json"), "r") as f:
        data = json.load(f)
        
    res = data.get("results", data)
    n_count = data.get("metadata", {}).get("N", 1000)
    
    # Format number with commas (e.g. 1000 -> 1{,}000)
    n_str = f"{n_count:,}".replace(",", "{,}")
        
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += f"\\caption{{Structural Trap SOTA Diagnostic ($N={n_str}$). Injecting the explicit structural harmonic via the Chromic Grid guarantees critical topological override across both convolutional and transformer-based discriminators. Unlike standard noise, structural grid metrics directly mirror the inherent clean baselines of standard unconstrained benchmarks.}}\n"
    tex += "\\label{tab:full_benchmark}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n"
    tex += "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    tex += "\\textbf{Target Model} & \\textbf{Clean Acc} & \\textbf{Natural ASR} & \\textbf{Grid ASR ($L$)} & \\textbf{Grid ASR ($L+C$)} \\\\ \\midrule\n"
    for k, v in res.items():
        clean = f"{v['Clean_Accuracy']*100:.1f}\\%"
        nat = f"{v['Natural_ASR']*100:.1f}\\%"
        gl = f"{v['Grid_L_ASR']*100:.1f}\\%"
        glc = f"{v['Grid_LC_ASR']*100:.1f}\\%"
        model_name = k.replace("_", "-")
        tex += f"{model_name} & {clean} & {nat} & {gl} & \\textbf{{{glc}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(SCRIPT_DIR, "table9_full_benchmark.tex"), "w") as f:
        f.write(tex)

if __name__ == '__main__':
    build_table5()
    build_table6()
    # build_table7()
    build_table9()
    print("Table generation complete.")
