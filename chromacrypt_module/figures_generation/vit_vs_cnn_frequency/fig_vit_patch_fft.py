import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os

def analyze_first_layer_embeddings():
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load Models
    print("Loading models...")
    try:
        vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(DEVICE)
        resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    except Exception:
        vit_model = models.vit_b_16(pretrained=True).to(DEVICE)
        resnet_model = models.resnet50(pretrained=True).to(DEVICE)
        
    # Extract first layer weights
    vit_weight = vit_model.conv_proj.weight.data.cpu() # Shape: [768, 3, 16, 16]
    resnet_weight = resnet_model.conv1.weight.data.cpu() # Shape: [64, 3, 7, 7]
    
    # We want to analyze the spatial frequency response
    # 1. Average across the input channels (RGB) to get spatial structure
    vit_spatial = vit_weight.mean(dim=1) # [768, 16, 16]
    resnet_spatial = resnet_weight.mean(dim=1) # [64, 7, 7]
    
    # 2. Perform 2D FFT on each filter
    vit_fft = torch.fft.fft2(vit_spatial, norm="ortho")
    resnet_fft = torch.fft.fft2(resnet_spatial, norm="ortho")
    
    # 3. Shift the zero-frequency component to the center of the spectrum
    vit_fft_shifted = torch.fft.fftshift(vit_fft, dim=(-2, -1))
    resnet_fft_shifted = torch.fft.fftshift(resnet_fft, dim=(-2, -1))
    
    # 4. Get magnitude spectrum and average across all filters
    vit_mag = torch.abs(vit_fft_shifted).mean(dim=0).numpy()
    resnet_mag = torch.abs(resnet_fft_shifted).mean(dim=0).numpy()
    
    # Plotting
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # ResNet Plot
    im0 = axes[0].imshow(resnet_mag, cmap='viridis', norm=LogNorm())
    axes[0].set_title('ResNet50 conv1\nMean Magnitude Spectrum')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # ViT Plot
    im1 = axes[1].imshow(vit_mag, cmap='viridis', norm=LogNorm())
    axes[1].set_title('ViT-B/16 Patch Embeddings\nMean Magnitude Spectrum')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), os.path.join(out_dir), 'fig_vit_patch_fft.png'), dpi=300)
    print(f"Saved to {os.path.join(out_dir, 'fig_vit_patch_fft.png')}")

if __name__ == "__main__":
    analyze_first_layer_embeddings()
