
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import chromacrypt_module as cc



def compute_saliency(model, image):
    model.eval()
    image.requires_grad_()
    
    # Canonical ImageNet Normalization before extracting forward activations natively.
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    norm_img = (image - mean) / std
    
    logits = model(norm_img.unsqueeze(0))
    score, _ = torch.max(logits, 1)
    score.backward()
    saliency, _ = torch.max(torch.abs(image.grad), dim=0)
    return saliency.detach().numpy()

def compute_fft(map2d):
    f = np.fft.fft2(map2d)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

def main():
    # Setup
    device = torch.device("cpu") # CPU is fine for single image
    
    # Models
    print("Loading ResNet50...")
    resnet = models.resnet50(pretrained=True).to(device)
    print("Loading ViT-B/16...")
    vit = models.vit_b_16(pretrained=True).to(device)
    
    # Dummy Image (Neutral Gray) + Chromic Interference Grid
    # We use a neutral background to isolate the grid's effect on saliency
    base_img = torch.ones(3, 224, 224) * 0.5
    grid = cc.generate_topological_grid(224, 224, device).squeeze(0) * 0.2
    attacked_img = (base_img + grid).clamp(0, 1).to(device)
    
    # Compute Saliency
    print("Computing ResNet Saliency...")
    sal_resnet = compute_saliency(resnet, attacked_img)
    
    # Reset gradients
    attacked_img.grad = None
    
    print("Computing ViT Saliency...")
    sal_vit = compute_saliency(vit, attacked_img)
    
    # Compute FFT
    fft_resnet = compute_fft(sal_resnet)
    fft_vit = compute_fft(sal_vit)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # ResNet
    axes[0, 0].imshow(sal_resnet, cmap='hot')
    axes[0, 0].set_title('ResNet50 Saliency (Attacked)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fft_resnet, cmap='inferno')
    axes[0, 1].set_title('ResNet50 Saliency FFT')
    axes[0, 1].axis('off')
    
    # ViT
    axes[1, 0].imshow(sal_vit, cmap='hot')
    axes[1, 0].set_title('ViT-B/16 Saliency (Attacked)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fft_vit, cmap='inferno')
    axes[1, 1].set_title('ViT-B/16 Saliency FFT')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "experiments", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "fig_vit_vs_cnn_fft.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved proof to {output_path}")

if __name__ == "__main__":
    main()
