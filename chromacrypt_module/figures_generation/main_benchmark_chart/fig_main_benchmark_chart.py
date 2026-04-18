import os

import os
import json
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results')

def generate_benchmark_figure():
    # Load dynamic metrics
    table9_path = os.path.join(RESULTS_DIR, "table9_full_benchmark.json")
    robust_path = os.path.join(RESULTS_DIR, "table10_autoattack_comparison.json")
    
    # Default fallbacks if missing cleanly natively
    r_grid, r_natural = 0.0, 0.0
    cn_grid, cn_natural = 0.0, 0.0
    sw_grid, sw_natural = 0.0, 0.0
    
    if os.path.exists(table9_path):
        with open(table9_path, 'r') as f:
            t9 = json.load(f)
            if "resnet50" in t9["results"]:
                r_grid = t9["results"]["resnet50"]["Grid_LC_ASR"] * 100
                r_natural = t9["results"]["resnet50"]["Natural_ASR"] * 100
                
    if os.path.exists(robust_path):
        with open(robust_path, 'r') as f:
            rob = json.load(f)
            if "results" in rob:
                for entry in rob["results"]:
                    if entry["model"] == "Liu2023Comprehensive_ConvNeXt-L":
                        cn_grid = entry["OKLCH_ASR"]
                        cn_natural = entry["RGB_ASR"]
                    if entry["model"] == "Xu2024MIMIR_Swin-L":
                        sw_grid = entry["OKLCH_ASR"]
                        sw_natural = entry["RGB_ASR"]

    models = ['ResNet50\n(Baseline)', 'Xu2024-Swin-L\n(Robust ViT)', 'Liu2023-ConvNeXt-L\n(Robust CNN)']
    
    # Attack Success Rates (ASR)
    grid_scores = [r_grid, sw_grid, cn_grid]
    natural_scores = [r_natural, sw_natural, cn_natural]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, grid_scores, width, label='Topological Attractor (Grid)', color='#e74c3c', alpha=0.9)
    rects2 = ax.bar(x + width/2, natural_scores, width, label='Narrowband FEATURE Mimicry', color='#3498db', alpha=0.9)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('SOTA Robustness Failure Benchmark (Continuous Empirical Validation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 120)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Bar labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    output_path = 'fig_full_benchmark.png'
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), output_path), dpi=300)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    generate_benchmark_figure()
