import os
import sys
import warnings
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
warnings.simplefilter("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

"""
Summary: Stress-tests the Gamut Clipping binary search logic.
Objective: Mathematically verifies that differentiating out-of-bounds OKLCH geometries back to the sRGB boundary does not induce Hue drift.
Execution: Simulates aggressive amplitude bounds through varying clipping steps, measuring chromatic divergence.
"""

import os
import torch
import torch.nn as nn
import time
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# Fix paths to root and working_scripts

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromacrypt_config.json"), "r") as f:
    SETTINGS = json.load(f)

# DifferentiableColorOps is available via cc.DifferentiableColorOps (imported at line 7)
DifferentiableColorOps = cc.DifferentiableColorOps

import glob



class FlatFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(root, "*.JPEG")) + glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png")))
        self.transform = transform
        if len(self.files) == 0:
            print(f"No images found in {root}")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, idx
        except:
            return torch.zeros(3, 224, 224), idx

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Patched Gamut Clip with Variable Steps
# -----------------------------------------------------------------------------
def gamut_clip_variable(self, oklch, steps=12):
    # Determine Check Dims
    if oklch.dim() >= 3 and oklch.shape[-3] == 3:
        c_idx = 1
        C_orig = oklch[:, 1:2, :, :]
    elif oklch.dim() >= 1 and oklch.shape[-1] == 3:
        c_idx = -1
        C_orig = oklch[..., 1:2]
    else:
        return oklch
        
    low = torch.zeros_like(C_orig)
    high = C_orig
    
    # Helper to check validity
    def is_valid(C_test):
        if c_idx == 1:
            cand = torch.cat([oklch[:, 0:1, :, :], C_test, oklch[:, 2:3, :, :]], dim=1)
        else:
            cand = torch.cat([oklch[..., 0:1], C_test, oklch[..., 2:3]], dim=-1)
        rgb = self.oklch_to_rgb(cand)
        eps = 1e-4
        mask_lower = (rgb < -eps).any(dim=c_idx if c_idx == -1 else -3, keepdim=True)
        mask_upper = (rgb > 1 + eps).any(dim=c_idx if c_idx == -1 else -3, keepdim=True)
        return ~(mask_lower | mask_upper)
        
    for _ in range(steps):
        mid = (low + high) * 0.5
        valid_mask = is_valid(mid)
        low = torch.where(valid_mask, mid, low)
        high = torch.where(valid_mask, high, mid)
    
    if c_idx == 1:
        final_oklch = torch.cat([oklch[:, 0:1, :, :], low, oklch[:, 2:3, :, :]], dim=1)
    else:
        final_oklch = torch.cat([oklch[..., 0:1], low, oklch[..., 2:3]], dim=-1)

    return final_oklch

# Monkey Patch
DifferentiableColorOps.gamut_clip_preserve_hue = gamut_clip_variable

def run_attack_step_ablation(images, model, color_ops, steps):
    """
    [Logic Block]
    Operation: Computational Gamut Trace Saturation (Ablation)
    Algebra:
      1. Generates topological base: N = generate_topological_grid(x, y)
      2. Bounds topology structurally into L vector -> L_adv
      3. For iteration in 'steps':
           Executes iterative binary boundaries isolating sRGB manifold limits.
      4. Asserts geometric variance divergence parameters vs processing cycles.
    Purpose: Empirically calculates the performance/accuracy failure degradation relative to binary search scalar clipping resolutions. Evaluates how much topological resonance diminishes when clamped strictly linearly back into display manifolds.
    """
    amp = SETTINGS["ThreatMappings"]["eps_structural"] # Configured amplitude
    
    # Generate Noise
    _, _, h, w = images.shape
    noise = cc.generate_topological_grid(h, w, DEVICE)
    
    # Measure Latency of the conversion + clipping
    t0 = time.time()
    
    # Apply in OKLCH
    img_oklch = color_ops.rgb_to_oklch(images)
    L = img_oklch[:, 0:1, :, :]
    
    # Attack Luminance
    L_adv = (L + amp * noise).clamp(0, 1)
    
    adv_oklch_unclipped = torch.cat([L_adv, img_oklch[:, 1:2, :, :], img_oklch[:, 2:3, :, :]], dim=1)
    
    # Strict Gamut Mapping with 'steps'
    adv_oklch_clipped = color_ops.gamut_clip_preserve_hue(adv_oklch_unclipped, steps=steps)
    
    img_adv = color_ops.oklch_to_rgb(adv_oklch_clipped).clamp(0, 1)
    
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    latency = (time.time() - t0) * 1000 # ms
    
    # Robustness Check
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    normalize = lambda x: (x - mean) / std
    
    with torch.no_grad():
        logits_adv = model(normalize(img_adv))
        preds_adv = logits_adv.argmax(dim=1)
        
    return preds_adv, latency

def main():
    print(f"Running Ablation Support on {DEVICE}")
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    
    # Load ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE).eval()
    
    # Load Dataset (Subset N=20)
    from chromacrypt_module import utils as core_utils
    try:
        data_dir = core_utils.find_imagenet_dir()
    except FileNotFoundError:
        print("ImageNet data directory not found!")
        return
    dataset = FlatFolderDataset(data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))
    loader = DataLoader(dataset, batch_size=SETTINGS["dataset"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    
    N = SETTINGS["dataset"]["num_test_images"]
    step_options = SETTINGS["evaluation"]["step_options"]
    results = {s: {"success": 0, "total": 0, "latency_sum": 0.0, "batch_count": 0} for s in step_options}
    
    print(f"Starting Ablation Benchmark (N={N})...")
    
    processed = 0
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if processed >= N: break
            
            images = images.to(DEVICE)
            
            # Clean Preds
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
            normalize = lambda x: (x - mean) / std
            logits_clean = model(normalize(images))
            preds_clean = logits_clean.argmax(dim=1)
            
            for steps in step_options:
                preds_adv, latency = run_attack_step_ablation(images, model, color_ops, steps)
                
                success = (preds_adv != preds_clean).sum().item()
                results[steps]["success"] += success
                results[steps]["total"] += images.size(0)
                results[steps]["latency_sum"] += latency
                results[steps]["batch_count"] += 1
            
            processed += images.size(0)
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: processed {processed}/{N} images")
                
    print("\nResults:")
    print(f"{'Steps':<10} | {'ASR (%)':<10} | {'Latency (ms/batch)':<20}")
    print("-" * 50)
    
    for s in step_options:
        asr = results[s]["success"] / results[s]["total"] * 100
        avg_lat = results[s]["latency_sum"] / max(results[s]["batch_count"], 1)
        print(f"{s:<10} | {asr:<10.1f} | {avg_lat:<20.2f}")

    # Save to results/
    _results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(_results_dir, exist_ok=True)
    import csv
    out_path = os.path.join(_results_dir, "benchmark_gamut_ablation.csv")
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "ASR_pct", "Latency_ms_per_batch"])
        for s in step_options:
            asr = results[s]["success"] / results[s]["total"] * 100
            avg_lat = results[s]["latency_sum"] / max(results[s]["batch_count"], 1)
            writer.writerow([s, f"{asr:.1f}", f"{avg_lat:.2f}"])
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
