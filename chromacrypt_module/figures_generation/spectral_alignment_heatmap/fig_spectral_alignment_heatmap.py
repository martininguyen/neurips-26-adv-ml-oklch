import os

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

def get_first_layer_weights(model):
    """Extracts the first layer weights (features) from a model."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m.weight.data.cpu().numpy()
    return None

def compute_fft_magnitude(image):
    """Computes the 2D FFT magnitude of an image."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

from chromacrypt_module.attacks import generate_topological_grid
def generate_chromic_interference_grid(size=224, A=0.5, T=4*np.pi):
    """Generates the Canonical Topological grid natively mapped from executed parameters."""
    return generate_topological_grid(size, size, 'cpu').squeeze().numpy()

def main():
    print("Loading models...")
    resnet = models.resnet50(pretrained=True)
    vgg = models.vgg16(pretrained=True)
    
    # 1. Extract Filter FFTs
    print("Extracting filters...")
    rn_filters = get_first_layer_weights(resnet) # [64, 3, 7, 7]
    vgg_filters = get_first_layer_weights(vgg)   # [64, 3, 3, 3]
    
    # We want to see the average frequency response of these filters
    # Pad to 224x224 to match the grid resolution for FFT comparison
    pad_h = 224
    pad_w = 224
    
    def get_avg_filter_fft(filters, size):
        accum_fft = np.zeros((pad_h, pad_w))
        count = 0
        
        # filters shape: [Out, In, H, W]
        # We'll take the luminance approximation (mean across RGB input channels)
        # or just sum all of them.
        
        num_filters = filters.shape[0]
        
        for i in range(num_filters):
            # Average across input channels (RGB) to get spatial structure
            kernel = np.mean(filters[i], axis=0) # [H, W]
            
            # Pad to image size
            kh, kw = kernel.shape
            padded = np.zeros((pad_h, pad_w))
            
            # Center the kernel
            start_y = (pad_h - kh) // 2
            start_x = (pad_w - kw) // 2
            padded[start_y:start_y+kh, start_x:start_x+kw] = kernel
            
            # Compute FFT
            fft_mag = compute_fft_magnitude(padded)
            accum_fft += fft_mag
            count += 1
            
        return accum_fft / count

    print("Computing ResNet50 Filter FFT...")
    rn_fft = get_avg_filter_fft(rn_filters, 224)
    
    print("Computing VGG16 Filter FFT...")
    vgg_fft = get_avg_filter_fft(vgg_filters, 224)
    
    # 2. Extract Chromic Interference FFT
    print("Computing Chromic Interference FFT...")
    chromic_interference_grid = generate_chromic_interference_grid(size=224)
    mj_fft = compute_fft_magnitude(chromic_interference_grid)
    
    # Plotting
    print("Generating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Threshold Chromic Interference to find peaks for overlay
    # Normalize Chromic Interference FFT to 0-1
    mj_norm = (mj_fft - mj_fft.min()) / (mj_fft.max() - mj_fft.min())
    # Find peaks (simple threshold)
    peaks = np.argwhere(mj_norm > 0.8)
    
    # Map row, col to spatial frequency coordinates [-0.5, 0.5]
    peak_x = (peaks[:, 1] - 112) / 224.0
    peak_y = (112 - peaks[:, 0]) / 224.0
    extent = [-0.5, 0.5, -0.5, 0.5]
    
    # Plot Chromic Interference FFT
    axes[0].imshow(mj_fft, cmap='inferno', extent=extent)
    axes[0].set_title("1. Attack Grid Frequencies", fontsize=14)
    axes[0].set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=12)
    axes[0].set_ylabel("Spatial Frequency (cycles/pixel)", fontsize=12)
    
    # Plot ResNet FFT with Overlay
    axes[1].imshow(rn_fft, cmap='inferno', extent=extent)
    # Overlay Chromic Interference peaks
    axes[1].scatter(peak_x, peak_y, color='cyan', s=20, alpha=0.8, label='Attack Peaks')
    axes[1].set_title("2. ResNet50 Sensitivity", fontsize=14)
    axes[1].set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=12)
    axes[1].set_ylabel("Spatial Frequency (cycles/pixel)", fontsize=12)
    # Add legend to explain dots
    axes[1].legend(loc='upper right')
    
    # Plot VGG FFT with Overlay
    axes[2].imshow(vgg_fft, cmap='inferno', extent=extent)
    axes[2].scatter(peak_x, peak_y, color='cyan', s=20, alpha=0.8, label='Attack Peaks')
    axes[2].set_title("3. VGG16 Sensitivity", fontsize=14)
    axes[2].set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=12)
    axes[2].set_ylabel("Spatial Frequency (cycles/pixel)", fontsize=12)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), 'fig_spectral_alignment.png'), dpi=150, bbox_inches='tight')
    print("Saved fig_spectral_alignment.png")

if __name__ == "__main__":
    main()
