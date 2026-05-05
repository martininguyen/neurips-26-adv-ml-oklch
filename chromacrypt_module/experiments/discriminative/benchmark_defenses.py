import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

"""
Summary: Evaluates OKLCH attacks against standard digital defenses.
Objective: Proves OKLCH grids bypass legacy pre-processing defenses (JPEG Compression, Total Variance Minimization) which traditionally destroy raw PGD static noise.
Execution: Chains PGD/OKLCH patterns through simulated defense pipelines and re-evaluates predictive confidence.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import torchattacks
import lpips
from skimage.restoration import denoise_tv_chambolle
from diffusers import AutoencoderKL
from chromacrypt_module.attacks import TopologicalAttractor
from tqdm import tqdm
import json

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromacrypt_config.json"), "r") as f:
    SETTINGS = json.load(f)

# Add project paths for core_utils and oklch_defense
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_working_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [_project_root, _working_scripts, os.path.join(_project_root, 'oklch_defense')]:
    if p not in sys.path:
        sys.path.append(p)

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Color ops for OKLCH conversions
color_ops = cc.DifferentiableColorOps().to(DEVICE)

# Load LPIPS Metric
loss_fn_lpips = lpips.LPIPS(net="vgg").to(DEVICE)

# 1. Load Threat Model (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()
model.to(DEVICE)

# Normalize transform expected by torchvision ResNet50
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 2. Load Defenses
print("Loading Defenses...")
# VAE -> SDE (Diffusion Flow-Matching)
from diffusers import StableDiffusionImg2ImgPipeline
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(DEVICE)
pipeline.set_progress_bar_config(disable=True)

def defend_jpeg(img_tensor, quality=50):
    """JPEG Compression via PIL"""
    img_pil = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    t = transforms.ToTensor()
    return t(compressed_pil).unsqueeze(0).to(DEVICE)

def defend_tv(img_tensor, weight=0.1):
    """Total Variation Denoising via skimage"""
    img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    denoised_np = denoise_tv_chambolle(img_np, weight=weight, channel_axis=2)
    denoised_tensor = torch.from_numpy(denoised_np).permute(2, 0, 1).unsqueeze(0).float()
    return denoised_tensor.to(DEVICE)

def defend_diffusion(img_tensor):
    """Autoencoder Purification via SD SDE Denoising Pipeline"""
    results_tensors = []
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    # Pipeline requires PIL images
    for i in range(img_tensor.shape[0]):
        img = to_pil(img_tensor[i].cpu())
        with torch.no_grad():
            purified_img = pipeline(prompt="", image=img, strength=0.35, guidance_scale=0.0).images[0]
        results_tensors.append(to_tensor(purified_img).to(DEVICE))
    return torch.stack(results_tensors)

# ---------------------------------------------------------------------------
# Data Loading (ImageNet-1K validation set)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
val_dir = os.path.join(os.path.dirname(_project_root), "data", "imagenet-1k")
if not os.path.exists(val_dir):
    val_dir = os.path.join(os.path.dirname(_project_root), "data", "datasets", "imagenet-1k")
if not os.path.exists(val_dir):
    print(f"Error: Could not find ImageNet-1K validation directory")
    sys.exit(1)

files = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG') or f.endswith('.jpg') or f.endswith('.png')])
if not files:
    print(f"Error: No image files found in {val_dir}")
    sys.exit(1)

sample_files = files[:SETTINGS["dataset"]["num_test_images"]]
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load real ground truth labels
_label_file = os.path.join(val_dir, "val_pytorch_labels.txt")
if os.path.exists(_label_file):
    with open(_label_file) as _f:
        _all_labels = [int(line.strip()) for line in _f.readlines()]
    print(f"Loaded {len(_all_labels)} real ground truth labels")
    USE_REAL_LABELS = True
else:
    _all_labels = None
    USE_REAL_LABELS = False
    print("WARNING: val_pytorch_labels.txt not found — using pseudo-labels")

# ---------------------------------------------------------------------------
# Main Loop Setup
# ---------------------------------------------------------------------------
results = {
    "Clean": {"lpips": [], "Nodefense_Succ": 0, "JPEG_Succ": 0, "TV_Succ": 0, "Diffusion_Succ": 0},
    "RGB PGD": {"lpips": [], "Nodefense_Succ": 0, "JPEG_Succ": 0, "TV_Succ": 0, "Diffusion_Succ": 0},
    "OKLCH PGD": {"lpips": [], "Nodefense_Succ": 0, "JPEG_Succ": 0, "TV_Succ": 0, "Diffusion_Succ": 0},
    "Luminance Grid (L)": {"lpips": [], "Nodefense_Succ": 0, "JPEG_Succ": 0, "TV_Succ": 0, "Diffusion_Succ": 0},
    "Chromic Interference (L+C)": {"lpips": [], "Nodefense_Succ": 0, "JPEG_Succ": 0, "TV_Succ": 0, "Diffusion_Succ": 0}
}
total_images = len(sample_files)

# Attack configurations
eps_base = SETTINGS["ThreatMappings"]["eps_rgb"]
alpha_base = eps_base / 4.0
steps_base = SETTINGS.get("ablation", {}).get("pgd_steps", 10)
atk_rgb = torchattacks.PGD(model, eps=eps_base, alpha=alpha_base, steps=steps_base)

def evaluate_image(x_tensor, true_label):
    """Returns True if the model incorrectly classifies (Adversarial Success), False if it's correct."""
    x_norm = normalize(x_tensor)
    with torch.no_grad():
        out = model(x_norm)
        pred = out.argmax(dim=1).item()
    return pred != true_label

for i, img_name in enumerate(sample_files):
    print(f"\\nProcessing Image {i+1}/{total_images}: {img_name}")
    img_path = os.path.join(val_dir, img_name)
    img_pil = Image.open(img_path).convert('RGB')
    
    # Base Image (224x224 already prepared for model input)
    x_clean = resize_transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # Get True Label (real ground truth or pseudo-label fallback)
    if USE_REAL_LABELS and _all_labels is not None:
        y_true = torch.tensor([_all_labels[i]], device=DEVICE)
    else:
        with torch.no_grad():
            y_true = model(normalize(x_clean)).argmax(dim=1)
        
    print(" -> Generating Attacks...")
    # 1. Clean (No Attack)
    x_clean_adv = x_clean.clone()
    
    # 2. Standard RGB PGD
    x_rgb_pgd = atk_rgb(x_clean, y_true).detach()
    
    # 3. OKLCH PGD (Stealthy) — via OKLCHModelWrapper + torchattacks
    oklch_wrapper = core_utils.OKLCHModelWrapper(model, freeze_H=True).to(DEVICE).eval()
    atk_oklch = torchattacks.PGD(oklch_wrapper, eps=SETTINGS["ThreatMappings"]["eps_oklch_l"], alpha=SETTINGS["ThreatMappings"]["eps_oklch_l"]/4.0, steps=steps_base)

    # Convert clean image to normalized OKLCH for the attack
    with torch.no_grad():
        x_oklch_raw = color_ops.rgb_to_oklch(x_clean)
        L_n = x_oklch_raw[:, 0:1]
        C_n = x_oklch_raw[:, 1:2] / 0.4
        H_n = x_oklch_raw[:, 2:3] / 360.0
        x_oklch_norm = torch.cat([L_n, C_n, H_n], dim=1).clamp(0, 1)
        oklch_wrapper.clean_oklch_norm = x_oklch_norm

    adv_oklch_norm = atk_oklch(x_oklch_norm, y_true).detach()

    # Convert back to RGB
    with torch.no_grad():
        adv_oklch = oklch_wrapper.unscale(adv_oklch_norm)
        x_oklch_pgd = color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(adv_oklch, steps=12)).clamp(0, 1)

    # 4. Chromic Interference (Universal Structural Grid)
    #    Evaluates localized topological resistance to explicit geometric manipulation via native continuous grid injection avoiding high L_inf spikes.
    atk_grid_l = TopologicalAttractor(eps=SETTINGS["ThreatMappings"]["eps_structural"], channel="L")
    atk_grid_lc = TopologicalAttractor(eps=SETTINGS["ThreatMappings"]["eps_structural"], channel="L+C")

    with torch.no_grad():
        x_chromic_l = atk_grid_l(x_clean, color_ops)
        x_chromic_lc = atk_grid_lc(x_clean, color_ops)

    attacks = {
        "Clean": x_clean_adv,
        "RGB PGD": x_rgb_pgd,
        "OKLCH PGD": x_oklch_pgd,
        "Luminance Grid (L)": x_chromic_l,
        "Chromic Interference (L+C)": x_chromic_lc
    }
    
    # Evaluate Pipeline
    for atk_name, x_adv in attacks.items():
        # LPIPS
        with torch.no_grad():
            img1 = x_clean * 2.0 - 1.0 
            img2 = x_adv * 2.0 - 1.0
            lpips_val = loss_fn_lpips(img1, img2).item()
        results[atk_name]["lpips"].append(lpips_val)

        # Base Success (No Defense)
        if evaluate_image(x_adv, y_true.item()):
            results[atk_name]["Nodefense_Succ"] += 1
            
        # JPEG Defense
        x_jpeg = defend_jpeg(x_adv, quality=50)
        if evaluate_image(x_jpeg, y_true.item()):
            results[atk_name]["JPEG_Succ"] += 1
            
        # TV Denoising Defense
        x_tv = defend_tv(x_adv, weight=0.1)
        if evaluate_image(x_tv, y_true.item()):
            results[atk_name]["TV_Succ"] += 1
            
        # Diffusion Autoencoder Defense
        x_vae = defend_diffusion(x_adv)
        if evaluate_image(x_vae, y_true.item()):
            results[atk_name]["Diffusion_Succ"] += 1

print("\\n" + "="*80)
print(f"{'Method':<22} | {'LPIPS':<8} | {'No Defense':<12} | {'JPEG (Q=50)':<12} | {'TV Denoise':<12} | {'SD 1.5 (SDE)'}")
print("-" * 80)
for atk_name, stats in results.items():
    avg_lpips = np.mean(stats['lpips'])
    nodef_succ = (stats['Nodefense_Succ'] / total_images) * 100
    jpeg_succ = (stats['JPEG_Succ'] / total_images) * 100
    tv_succ = (stats['TV_Succ'] / total_images) * 100
    vae_succ = (stats['Diffusion_Succ'] / total_images) * 100
    print(f"{atk_name:<22} | {avg_lpips:<8.4f} | {nodef_succ:<11.1f}% | {jpeg_succ:<11.1f}% | {tv_succ:<11.1f}% | {vae_succ:<11.1f}%")
print("="*80)

# Save results
out_data = {
    "total_images": total_images,
    "results": results
}
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
try:
    with open(os.path.join(out_dir, "benchmark_defenses_output.json"), "w") as f:
        json.dump(out_data, f, indent=4)
except Exception as e:
    print(f"JSON Dump Serialization Error: {e}")

# Explicit CUDA/Xformers Teardown to prevent core dumps on script completion
import gc
try:
    del pipeline
    del model
    del atk_rgb
    del atk_oklch
    del oklch_wrapper
except: 
    pass
gc.collect()
torch.cuda.empty_cache()
print("Teardown completion successfully enforced - shutting down.")
