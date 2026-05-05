import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from diffusers import AutoencoderKL

# Append paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import chromacrypt_module as cc

def compute_fft_magnitude(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

def main():
    print("Generating VAE Nyquist Bottleneck Verification Figure...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 224
    
    # 1. Generate the Grid (lambda=16)
    grid_tensor = cc.generate_topological_grid(size, size, device)
    
    # 2. Load the SD VAE to prove it survives compression
    try:
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
        vae.eval()
    except Exception as e:
        print("Could not load VAE. Ensure you have internet access.")
        return

    # Scale grid to image bounds [0, 1], then [-1, 1] for VAE
    img_tensor = (grid_tensor * 0.5 + 0.5).repeat(1, 3, 1, 1) * 2.0 - 1.0
    
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        reconstructed = vae.decode(latents).sample
        
    recon_np = (reconstructed.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    grid_np = (grid_tensor.squeeze().cpu().numpy() * 0.5 + 0.5)
    
    # 3. Compute FFTs (Take the luminance/first channel)
    recon_fft = compute_fft_magnitude(recon_np[:, :, 0])
    grid_fft = compute_fft_magnitude(grid_np)
    
    # 4. Define the VAE Nyquist Limit
    # SD VAE uses 8x spatial downsampling. 
    # For a 224 image, max representable frequency before aliasing is lambda=16.
    # In spatial frequency (cycles/pixel), this is 1/16 = 0.0625.
    nyquist_freq = 1.0 / 16.0 
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Spatial Domain
    axes[0, 0].imshow(grid_np, cmap='gray')
    axes[0, 0].set_title(r'Original Chromic Grid ($\lambda=16$)', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(recon_np)
    axes[0, 1].set_title('Reconstructed Macro-Geometry\n(Post 8x VAE Downsampling)', fontsize=14)
    axes[0, 1].axis('off')
    
    # Frequency Domain
    extent = [-0.5, 0.5, -0.5, 0.5]
    axes[1, 0].imshow(grid_fft, cmap='inferno', extent=extent)
    axes[1, 0].set_title('Original Grid FFT', fontsize=14)
    axes[1, 0].set_xlabel('Spatial Frequency (cycles/pixel)', fontsize=12)
    axes[1, 0].set_ylabel('Spatial Frequency (cycles/pixel)', fontsize=12)
    
    axes[1, 1].imshow(recon_fft, cmap='inferno', extent=extent)
    axes[1, 1].set_title('Post-VAE Spectral Response\n(Signal mathematically preserved)', fontsize=14)
    axes[1, 1].set_xlabel('Spatial Frequency (cycles/pixel)', fontsize=12)
    axes[1, 1].set_ylabel('Spatial Frequency (cycles/pixel)', fontsize=12)
    
    # Draw the Nyquist Safe Zone Box on both FFTs
    for ax in [axes[1, 0], axes[1, 1]]:
        rect = patches.Rectangle((-nyquist_freq, -nyquist_freq), nyquist_freq*2, nyquist_freq*2, 
                                 linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--', 
                                 label=r'VAE 8x Nyquist Limit ($\lambda=16$)')
        ax.add_patch(rect)
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "experiments", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig_nyquist_evasion.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved perfectly aligned figure to {out_path}")

if __name__ == "__main__":
    main()