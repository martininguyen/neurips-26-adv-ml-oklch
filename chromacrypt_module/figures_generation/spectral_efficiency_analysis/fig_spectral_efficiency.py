import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

def generate_grid(size=224, period=4.0):
    """Generates a 2D Chromic Interference sine wave grid"""
    x = torch.arange(size, dtype=torch.float32)
    y = torch.arange(size, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.sin(2 * torch.pi * X / period) * torch.sin(2 * torch.pi * Y / period)
    # Replicate to 3 channels: (1, 3, H, W)
    grid = grid.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return grid

def generate_noise(size=224):
    """Generates Uniform Random Noise in [-1, 1]"""
    return (torch.rand(1, 3, size, size) * 2 - 1)

def compute_filter_rspectral(model):
    """Extracts first conv layer and computes the activation ratio per filter"""
    conv1 = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv1 = m
            break
            
    if not conv1:
        return np.ones(64)
        
    num_filters = conv1.out_channels
    
    # Use deterministic grid with optimal period (approx 4.0-6.0 for ResNet 7x7 kernels)
    grid = generate_grid(size=224, period=5.5)
    
    # Compute Expected absolute activation for Grid per filter
    with torch.no_grad():
        # Shape: (1, num_filters, H, W) -> mean over spatial and batch (0, 2, 3)
        act_grid = conv1(grid).abs().mean(dim=(0, 2, 3)).numpy()
        
    # Compute Expected absolute activation for Noise per filter
    act_noise_sum = np.zeros(num_filters)
    num_samples = 50
    for _ in range(num_samples):
        noise = generate_noise(size=224)
        with torch.no_grad():
            act_noise_sum += conv1(noise).abs().mean(dim=(0, 2, 3)).numpy()
            
    act_noise = act_noise_sum / num_samples
    
    # Compute Ratio per filter
    # Add epsilon to prevent division by zero
    r_spectral = act_grid / (act_noise + 1e-8)
    return r_spectral

def main():
    print("Loading models...")
    # Pretrained models
    rn50_pre = models.resnet50(pretrained=True)
    vgg16_pre = models.vgg16(pretrained=True)
    
    # Randomly initialized models
    rn50_rand = models.resnet50(pretrained=False)
    vgg16_rand = models.vgg16(pretrained=False)
    
    print("Computing Rspectral metrics per filter...")
    rn50_pre_r = compute_filter_rspectral(rn50_pre)
    rn50_rand_r = compute_filter_rspectral(rn50_rand)
    
    vgg16_pre_r = compute_filter_rspectral(vgg16_pre)
    vgg16_rand_r = compute_filter_rspectral(vgg16_rand)
    
    # Prepare data for Seaborn Violin plot
    data = []
    
    for val in rn50_pre_r:
        data.append({'Model': 'ResNet50\n(Pretrained)', 'R_spectral': val})
    for val in rn50_rand_r:
        data.append({'Model': 'ResNet50\n(Randomly Init)', 'R_spectral': val})
        
    for val in vgg16_pre_r:
        data.append({'Model': 'VGG16\n(Pretrained)', 'R_spectral': val})
    for val in vgg16_rand_r:
        data.append({'Model': 'VGG16\n(Randomly Init)', 'R_spectral': val})
        
    df = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Use a visually distinct color palette
    palette = {"ResNet50\n(Pretrained)": "#1f77b4", 
               "ResNet50\n(Randomly Init)": "#aec7e8", 
               "VGG16\n(Pretrained)": "#ff7f0e", 
               "VGG16\n(Randomly Init)": "#ffbb78"}
               
    # Create violin plot
    sns.violinplot(x='Model', y='R_spectral', data=df, palette=palette, inner='quartile', linewidth=1.5)
    
    # Overlay swarm plot for showing individual standard filter points
    sns.swarmplot(x='Model', y='R_spectral', data=df, color='k', alpha=0.5, size=3)
    
    # Add a dashed line at y=1.0 representing natural noise baseline
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Random Noise Baseline ($R_{spectral} = 1.0$)')
    
    plt.ylabel('Filter-wise Relative Activation Ratio ($R_{spectral}$)', fontsize=14)
    plt.title('Distribution of Spectral Efficiency ($R_{spectral}$) Across First-Layer Filters', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), 'fig_attack_spectrum.png'), dpi=300)
    print("Saved -> fig_attack_spectrum.png")
    
if __name__ == "__main__":
    main()
