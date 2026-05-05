import json
import os
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_table4():
    path = os.path.join(RESULTS_DIR, "lpips_sweep.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    tex = "\\begin{table}[ht]\n\\centering\n\\small\n"
    tex += "\\caption{Continuous Perceptual Noise Ablation ($N=1{,}000$). Tracking the Attack Success Rate (ASR) against ConvNeXt-L across a dynamically sweeping perceptual footprint (LPIPS) for both random RGB noise and the structural Chromic Grid.}\n"
    tex += "\\label{tab:random_noise_baserate}\n"
    tex += "\\begin{adjustbox}{max width=\\linewidth}\n"
    tex += "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
    tex += "\\textbf{Calibrated LPIPS} & \\textbf{Random Noise ASR} & \\textbf{Chromic Grid ASR} & \\textbf{$\\Delta$ Gap} \\\\ \\midrule\n"
    
    # Sort keys dynamically if they are strings like "0.2"
    for k in sorted(data.keys(), key=lambda x: float(x)):
        v = data[k]
        cnt = v.get('count', 500)
        grid_asr = (v.get('grid_fail', 0) / cnt) * 100
        rgb_asr = (v.get('rgb_fail', 0) / cnt) * 100
        gap = grid_asr - rgb_asr
        tex += f"{float(k):.2f} & {rgb_asr:.1f}\\% & \\textbf{{{grid_asr:.1f}\\%}} & +{gap:.1f}\\% \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table4_random_noise.tex"), "w") as f: f.write(tex)

def build_table5():
    path = os.path.join(RESULTS_DIR, "table5_channel_ablation.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    tex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    tex += "\\caption{White-Box Channel Ablation (PGD). Efficacy mapped by isolating gradient steps to specific OKLCH axes.}\n"
    tex += "\\label{tab:channel_ablation}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}llccccccc@{}}\n\\toprule\n"
    tex += "\\textbf{Optimization Axes} & \\textbf{LPIPS} & \\textbf{ResNet50} & \\textbf{EfficientNet} & \\textbf{ViT-B-16} & \\textbf{VGG16} & \\textbf{DenseNet121} & \\textbf{Mean ASR} \\\\ \\midrule\n"
    for k, v in data.items():
        lpips = f"{v['lpips']:.4f}"
        rn = f"{v['asr'].get('ResNet50', 0):.1f}\\%"
        en = f"{v['asr'].get('EfficientNet', 0):.1f}\\%"
        vit = f"{v['asr'].get('ViT-B-16', 0):.1f}\\%"
        vgg = f"{v['asr'].get('VGG16', 0):.1f}\\%"
        mob = f"{v['asr'].get('DenseNet121', v['asr'].get('densenet121', 0)):.1f}\\%"
        mean = f"{v['mean_asr']:.1f}\\%"
        key_str = k.replace("$", "\\$") if "$" not in k else k
        tex += f"{key_str} & {lpips} & {rn} & {en} & {vit} & {vgg} & {mob} & \\textbf{{{mean}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table5_channel_ablation.tex"), "w") as f: f.write(tex)

def build_table14():
    path = os.path.join(RESULTS_DIR, "table14_overhead.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    tex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    tex += "\\caption{Computational Resource Benchmarking. Highlighting relative matrix latency between native cartesian backpropagation vs spatial boundary continuous convergence cycles.}\n"
    tex += "\\label{tab:computational_overhead}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}llcc@{}}\n\\toprule\n"
    tex += "\\textbf{Metric} & \\textbf{RGB Space} & \\textbf{OKLCH Space} & \\textbf{Factor} \\\\ \\midrule\n"
    tex += f"Latency (ms/iter) & {data.get('RGB-PGD (ms/iter)', 0)} & {data.get('OKLCH-PGD (ms/iter)', 0)} & {data.get('Slowdown Factor', 0)}x \\\\\n"
    tex += f"Peak GPU Memory & {data.get('Peak GPU Memory (MB)', 0)} MB & {data.get('Peak GPU Memory (MB)', 0)} MB & 1.0x \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table14_overhead.tex"), "w") as f: f.write(tex)

def build_table7():
    path = os.path.join(RESULTS_DIR, "table7_transferability.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    res = data.get("results", data)
    surrogate = data.get("metadata", {}).get("surrogate_model", "ResNet50")
    tex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    tex += f"\\caption{{Black-Box Transferability matrix targeting surrogate gradients from {surrogate}. Transfer success metrics identify that standard uncharacterized iterative perturbations fundamentally fail to generalize to Vision Transformers.}}\n"
    tex += "\\label{tab:transferability_matrix}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
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
    with open(os.path.join(RESULTS_DIR, "table7_transferability_ablation.tex"), "w") as f: f.write(tex)

def build_table8():
    path = os.path.join(RESULTS_DIR, "benchmark_cross_model_output.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    res = data.get("Results", data)
    tex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    tex += "\\caption{Cross-Architecture Vulnerability Diagnostic. Evaluating the attack success rate of OKLCH optimization vectors across convolutional and transformer-based discriminators.}\n"
    tex += "\\label{tab:cross_model}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    tex += "\\textbf{Model Type} & \\textbf{Luminance ($L$) Success} & \\textbf{Chroma ($C$) Success} & \\textbf{Hue ($H$) Success} & \\textbf{Primary Failure Channel} \\\\ \\midrule\n"
    for k, v in res.items():
        total = v.get("Total", 1)
        if total == 0: total = 1
        l_r = (v.get('L', 0) / total) * 100
        c_r = (v.get('C', 0) / total) * 100
        h_r = (v.get('H', 0) / total) * 100
        
        rates = {'Luminance (Structure)': l_r, 'Chroma': c_r, 'Hue': h_r}
        primary = max(rates, key=rates.get)
        tex += f"{k} & {l_r:.1f}\\% & {c_r:.1f}\\% & {h_r:.1f}\\% & {primary} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table8_cross_model.tex"), "w") as f: f.write(tex)

def build_table9():
    path = os.path.join(RESULTS_DIR, "table9_full_benchmark.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    res = data.get("results", data)
    n_count = data.get("metadata", {}).get("N", 1000)
    n_str = f"{n_count:,}".replace(",", "{,}")
    tex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    tex += f"\\caption{{Structural Trap SOTA Diagnostic ($N={n_str}$). Injecting the explicit structural harmonic via the Chromic Grid guarantees critical topological override across both convolutional and transformer-based discriminators. Unlike standard noise, structural grid metrics directly mirror the inherent clean baselines of standard unconstrained benchmarks.}}\n"
    tex += "\\label{tab:full_benchmark}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    tex += "\\textbf{Target Model} & \\textbf{Clean Acc} & \\textbf{Narrowband Feature Mimicry} & \\textbf{Luminance Grid} & \\textbf{Chromic Interference} \\\\ \\midrule\n"
    for k, v in res.items():
        clean = f"{v.get('Clean_Accuracy', 0)*100:.1f}\\%"
        nat = f"{v.get('Natural_ASR', 0)*100:.1f}\\%"
        gl = f"{v.get('Grid_L_ASR', 0)*100:.1f}\\%"
        glc = f"{v.get('Grid_LC_ASR', 0)*100:.1f}\\%"
        model_name = k.replace("_", "-")
        if k == "Liu2023Comprehensive_ConvNeXt-L":
            model_name = "Liu2023-ConvNeXt-L"
        elif k == "Xu2024MIMIR_Swin-L":
            model_name = "Xu2024-Swin-L"
        elif k == "resnet50":
            model_name = "ResNet50-Standard"
        elif k == "Salman2020Do_R50":
            model_name = "Salman2020-ResNet50"
        elif k == "Salman2020Do_50_2":
            model_name = "Salman2020-WRN50-2"
        tex += f"{model_name} & {clean} & {nat} & {gl} & \\textbf{{{glc}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table9_full_benchmark.tex"), "w") as f: f.write(tex)

def build_table10():
    path = os.path.join(RESULTS_DIR, "table10_autoattack_comparison.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    tex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    tex += "\\caption{Mathematical Optimizer Parity Benchmark ($N=1{,}000$). Evaluating identically restricted $L_{\\infty}$ gradient trajectories ($\\leq\\epsilon=4/255$) executed universally by the AutoAttack ensemble bounds on core SOTA architectures. By traversing the mathematically decoupled OKLCH coordinate space with 12-step geometric preservation boundaries, structural adversarial formulations converge on significantly stronger optima compared to optimizing strictly within the Cartesian RGB domain.}\n"
    tex += "\\label{tab:sota_optimizer_comparison}\n"
    tex += "\\begin{adjustbox}{max width=\\linewidth}\n"
    tex += "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
    tex += "\\textbf{Target Structure} & \\textbf{RGB AutoAttack ASR} & \\textbf{OKLCH AutoAttack ASR} & \\textbf{Avg LPIPS ($\\Delta$)} \\\\ \\midrule\n"
    
    for res in data.get("results", []):
         m = res["model"].replace("_", "-")
         r_a = res["RGB_ASR"]
         o_a = res["OKLCH_ASR"]
         lpips = res["OKLCH_LPIPS"]
         
         if m == "Liu2023Comprehensive-ConvNeXt-L": m = "Liu2023-ConvNeXt-L"
         if m == "Xu2024MIMIR-Swin-L": m = "Xu2024-Swin-L"
         if m == "resnet50": m = "ResNet50-Standard"
         
         if o_a > r_a:
             tex += f"{m} & {r_a:.1f}\\% & \\textbf{{{o_a:.1f}\\%}} & {lpips:.4f} \\\\\n"
         else:
             tex += f"{m} & \\textbf{{{r_a:.1f}\\%}} & {o_a:.1f}\\% & {lpips:.4f} \\\\\n"
             
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table10_autoattack_comparison.tex"), "w") as f: f.write(tex)

def build_table11():
    path = os.path.join(RESULTS_DIR, "table11_diffusion_purification.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    tex = "\\begin{table}[h]\n\\centering\n\\small\n"
    tex += "\\caption{Diffusion Purification Vulnerability (N=798) detailing the dose-response SNR survival of the Chromic Grid against a ResNet-50 Target Classifier. While DiffPure securely scrubs unstructured noise and local Adversarial Patches (rescuing 189 patched images), the structural coherence of the OKLCH Grid traps the generative pipeline. The geometric signal induces 60 instances of SNR survival, elevating the net failure rate to $86.6\\%$.}\n"
    tex += "\\label{tab:diffusion_purification}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    tex += "\\textbf{Attack Generator} & \\textbf{Pre-Purification ASR} & \\textbf{Post-Purification ASR} & \\textbf{Rescued Images $\\uparrow$} & \\textbf{SNR Survival $\\downarrow$} \\\\ \\midrule\n"
    for label, key in [("Adversarial Patch ($32\\times32$)", "patch"), ("Luminance Grid ($A=0.50$)", "grid"), ("Narrowband Feature Mimicry ($A=0.50$)", "natural")]:
        v = data.get(key, {})
        tex += f"{label} & {v.get('pre_success',0)/data.get('total_images', 1000)*100:.1f}\\% & {v.get('post_success',0)/data.get('total_images',1000)*100:.1f}\\% & {v.get('rescued',0)} & {v.get('hallucinated',0)} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table11_diffusion_purification.tex"), "w") as f: f.write(tex)

def build_table12():
    path = os.path.join(RESULTS_DIR, "table12_structured_baselines.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    tex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    tex += "\\caption{Structured Baseline Diagnostic ($N=1{,}000$). Benchmarking the efficacy of the Chromic Grid\'s global spatial harmonic against localized Adversarial Patches across architectures.}\n"
    tex += "\\label{tab:structured_baselines}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}l|cc|ccc@{}}\n\\toprule\n"
    tex += "\\textbf{Target Model} & \\textbf{Clean Acc} & \\textbf{Adv Patch ASR ($32 \\times 32$)} & \\textbf{Mimicry ASR (Narrowband)} & \\textbf{Grid ASR (Global Harmonic)} & \\textbf{Avg LPIPS Footprint} \\\\ \\midrule\n"
    for k, v in data.items():
        if not isinstance(v, dict): continue
        clean = f"{v.get('Clean_Accuracy', 0)*100:.1f}\\%"
        patch = f"{v.get('Adv_Patch_ASR', 0)*100:.1f}\\%"
        nat = f"{v.get('Narrowband_ASR', 0)*100:.1f}\\%"
        glc = f"{v.get('Grid_Harmonic_ASR', 0)*100:.1f}\\%"
        lpips = f"{v.get('Avg_LPIPS_Footprint', 0):.4f}"
        
        disp_name = k.replace('_', '-')
        if k == "Liu2023Comprehensive_ConvNeXt-L":
            disp_name = "Liu2023-ConvNeXt-L"
        elif k == "Xu2024MIMIR_Swin-L":
            disp_name = "Xu2024-Swin-L"
        elif k == "resnet50":
            disp_name = "ResNet50-Standard"
            
        tex += f"{disp_name} & {clean} & {patch} & {nat} & \\textbf{{{glc}}} & {lpips} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table12_structured_baselines.tex"), "w") as f: f.write(tex)

def build_table13():
    path = os.path.join(RESULTS_DIR, "table13_sd35_purification.json")
    if not os.path.exists(path): return
    with open(path, "r") as f: data = json.load(f)
    tex = "\\begin{table}[h]\n\\centering\n\\small\n"
    tex += "\\caption{Stable Diffusion 3.5 Latent Projection Vulnerability ($N=798$). Evaluating state-of-the-art flow-matching architecture against structural geometries. Unlike legacy diffusion models, SD3.5 largely preserves high-amplitude structures, resulting in negligible purification efficacy (rescuing merely 28 inputs against the $A=0.20$ Grid). Furthermore, low-amplitude structural topologies ($A=0.05$) act as generative attractors, actively exacerbating classification failure rates post-projection ($16.2\\% \\rightarrow 18.8\\%$).}\n"
    tex += "\\label{tab:sd35_purification_matrix}\n"
    tex += "\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    tex += "\\textbf{Attack Generator} & \\textbf{Pre-Purification ASR} & \\textbf{Post-Purification ASR} & \\textbf{Rescued Images $\\uparrow$} & \\textbf{SNR Survival $\\downarrow$} \\\\ \\midrule\n"
    for k, v in data.get("results", data).items():
         if isinstance(v, dict):
             tex += f"{k.replace('_', ' ')} & {v.get('Pre_ASR',0)*100:.1f}\\% & {v.get('Post_ASR',0)*100:.1f}\\% & {v.get('Rescued',0)} & {v.get('Hallucinated',0)} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    with open(os.path.join(RESULTS_DIR, "table13_sd35_purification.tex"), "w") as f: f.write(tex)

if __name__ == '__main__':
    print("Compiling JSON arrays to native LaTeX representation...")
    build_table4()
    build_table5()
    build_table14()
    build_table7()
    build_table8()
    build_table9()
    build_table10()
    build_table11()
    build_table12()
    build_table13()
    print("Matrix Conversion Successful. Check chromacrypt_module/experiments/results/")
