import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

"""
Summary: Evaluates channel fragmentation across diverse architectures.
Objective: Verifies if the hierarchy of channel vulnerability (e.g., L > C > H) remains consistent globally across convolutions (ResNet) vs transformers (ViT).
Execution: Iterates specific intense channel perturbations globally over a small targeted batch.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import glob
import csv
import numpy as np
import json

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromacrypt_config.json"), "r") as f:
    SETTINGS = json.load(f)

# Fix paths to root and working_scripts

# Import Differentiable Color Ops
from chromacrypt_module.color_ops import DifferentiableColorOps
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available(): DEVICE = "cuda"

def load_models():
    print("Loading Models...")
    nets = {
        "ResNet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE).eval(),
        "EfficientNet": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(DEVICE).eval(),
        "ViT": models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(DEVICE).eval(),
        "VGG16": models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(DEVICE).eval()
    }
    return nets

def predict(model, img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    norm = (img_tensor - mean) / std
    return model(norm).argmax(1).item()

def run_benchmark():
    models_dict = load_models()
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    
    # Data Setup
    img_dir = "../data/imagenet-1k"
    if not os.path.exists(img_dir):
        img_dir = "data/imagenet-1k" # Fallback
        
    all_files = sorted(glob.glob(os.path.join(img_dir, "*.JPEG")))
    if not all_files:
        print("No images found.")
        return

    # Configuration
    N = SETTINGS["dataset"]["num_test_images"]
    print(f"Benchmarking on {N} images...")
    
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Store results: [Model][Channel] = #Success
    stats = {m: {'L': 0, 'C': 0, 'H': 0, 'Total': 0} for m in models_dict.keys()}
    
    for i, path in enumerate(all_files):
        if i >= N: break
        
        try:
            img_pil = Image.open(path).convert('RGB')
            img_tensor = t(img_pil).unsqueeze(0).to(DEVICE)
            b, c, h, w = img_tensor.shape
            grid = cc.generate_topological_grid(h, w, DEVICE)
            
            # Pre-calc OKLCH
            oklch = color_ops.rgb_to_oklch(img_tensor)
            L = oklch[:, 0:1, :, :]
            C = oklch[:, 1:2, :, :]
            H_ch = oklch[:, 2:3, :, :]
            
            # Load attack parameters from config
            l_int = SETTINGS.get("attack", {}).get("eps_structural", 0.2)
            c_int = SETTINGS.get("attack", {}).get("eps_chromic", 8.0) / 100.0  # Normalized to 0.08
            h_int = 45.0  # Geometrically scaled phase rotation fallback
            
            # --- ATTACKS ---
            
            """
            [Logic Block]
            Operation: Disjoint Topological Channel Evaluation
            Algebra:
              1. L-Attack (Structural): Projects epsilon topology exclusively against geometric luminance layer.
                 L_mod = L + (epsilon_L * grid)
              2. C-Attack (Chromic): Injects uniform topological expansion radially against raw saturation arrays without altering boundary geometric contours.
                 C_mod = abs(C + (epsilon_C * grid))
              3. H-Attack (Hue): Mathematically scales rotational variance cylindrically without violating boundary gamut intensity.
                 H_mod = (H + h_int * grid) % 360
            Purpose: Identifies isolated failure vectors natively intrinsic to distinct dimensions of topological perception independent of scalar color values.
            """
            # 1. L-Attack (Structural Interference)
            L_mod = (L + l_int * grid) # L is 0-1 usually
            inv_L = torch.cat([L_mod, C, H_ch], dim=1)
            img_L = color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(inv_L, steps=12)).clamp(0, 1)
            
            # 2. C-Attack (Chromic Grid)
            C_mod = torch.abs(C + c_int * grid)
            inv_C = torch.cat([L, C_mod, H_ch], dim=1)
            img_C = color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(inv_C, steps=12)).clamp(0, 1)
            
            # 3. H-Attack (Hue Rotation/Shift)
            H_mod = (H_ch + h_int * grid) % 360.0
            inv_H = torch.cat([L, C, H_mod], dim=1)
            img_H = color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(inv_H, steps=12)).clamp(0, 1)
            
            # Evaluate per model
            for m_name, net in models_dict.items():
                clean_pred = predict(net, img_tensor)
                stats[m_name]['Total'] += 1
                
                # Check L
                if predict(net, img_L) != clean_pred:
                    stats[m_name]['L'] += 1
                
                # Check C
                if predict(net, img_C) != clean_pred:
                    stats[m_name]['C'] += 1
                    
                # Check H
                if predict(net, img_H) != clean_pred:
                    stats[m_name]['H'] += 1
                    
        except Exception as e:
            print(f"Err {i}: {e}")
            
        if i % 10 == 0: print(f"Progress {i}/{N}...")

    # Print Markdown Table
    print("\n\n### Model Failure Analysis")
    print("| Model Type | Chromic Interference (L) Success | Chroma (C) Success | Hue (H) Success | Primary Failure Channel |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for m_name, data in stats.items():
        total = data['Total']
        if total == 0: continue
        
        l_rate = data['L'] / total
        c_rate = data['C'] / total
        h_rate = data['H'] / total
        
        # Determine Primary
        rates = {'Luminance (Structure)': l_rate, 'Chroma': c_rate, 'Hue': h_rate}
        primary = max(rates, key=rates.get)
        
        print(f"| {m_name} | {l_rate:.1%} | {c_rate:.1%} | {h_rate:.1%} | {primary} |")
        
    out_data = {
        "Total_Images": N,
        "Results": stats
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark_cross_model_output.json"), "w") as f:
        json.dump(out_data, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
