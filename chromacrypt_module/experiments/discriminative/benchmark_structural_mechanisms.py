import torch
import torchvision.models as models
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import lpips

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "chromacrypt_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    num_images = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    eps_s = config["ThreatMappings"]["eps_structural"]
    
    print(f"Initializing ChromaCrypt Structural Matrix (N={num_images}, eps={eps_s})...")
    model = models.resnet50(pretrained=True).eval().to(DEVICE)
    loss_fn = lpips.LPIPS(net="alex").to(DEVICE)
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    
    """
    [Logic Block]
    Operation: Topological Spectral Disruption Vectors
    Algebra:
      1. NarrowbandMimicry: Executes mathematically constrained high-frequency spectral clustering mapped perfectly onto structural vulnerability zones (edges, textures).
      2. TopologicalAttractor: Generates explicit generalized baseline structural perturbations mathematically mapping universal geometrical collapse modes.
    Purpose: Compares strictly defined boundary logic (Targeted high-frequency gradients vs canonical geometric constraints) natively separating visual phenomena vulnerabilities. 
    """
    narrowband_atk = cc.NarrowbandMimicry(eps=eps_s, freq_mult=2.0)
    topo_atk = cc.TopologicalAttractor(eps=eps_s)
    patch_atk = cc.AdvPatch(model=model, patch_size=config["purification"]["patch_size"])
    
    total_imgs = 0
    narrow_wins = 0
    topo_wins = 0
    patch_wins = 0
    total_lpips_n = 0.0
    total_lpips_t = 0.0
    total_lpips_p = 0.0

    print("Executing iterative memory-safe evaluation loop...")
    for offset in range(0, num_images, batch_size):
        curr_batch_size = min(batch_size, num_images - offset)
        images, labels, mean, std = core_utils.load_imagenet_val_batch(n_examples=curr_batch_size, offset=offset)
        img_tensor = images.to(DEVICE)
        lbl_tensor = labels.to(DEVICE)
        
        # 1. Narrowband Mimicry Pass
        adv_n = narrowband_atk(img_tensor, color_ops)
        with torch.no_grad():
            preds_n = model((adv_n - mean) / std).argmax(1)
            narrow_wins += (preds_n != lbl_tensor).float().sum().item()
            total_lpips_n += loss_fn(img_tensor, adv_n).sum().item()
            
        # 2. Luminance Grid Pass
        adv_t = topo_atk(img_tensor, color_ops)
        with torch.no_grad():
            preds_t = model((adv_t - mean) / std).argmax(1)
            topo_wins += (preds_t != lbl_tensor).float().sum().item()
            total_lpips_t += loss_fn(img_tensor, adv_t).sum().item()
            
        # 3. Adversarial Patch Pass
        adv_p = patch_atk(img_tensor, labels=lbl_tensor)
        with torch.no_grad():
            preds_p = model((adv_p - mean) / std).argmax(1)
            patch_wins += (preds_p != lbl_tensor).float().sum().item()
            total_lpips_p += loss_fn(img_tensor, adv_p).sum().item()
        
        total_imgs += curr_batch_size
        print(f"  -> Processed {total_imgs}/{num_images} matrix boundaries")

    asr_n = (narrow_wins/total_imgs)*100
    asr_t = (topo_wins/total_imgs)*100
    asr_p = (patch_wins/total_imgs)*100
    lpips_n = total_lpips_n/total_imgs
    lpips_t = total_lpips_t/total_imgs
    lpips_p = total_lpips_p/total_imgs

    print(f"\n[Final Narrowband Feature Mimicry Mechanics]")
    print(f"ASR: {asr_n:.1f}% | Metric LPIPS: {lpips_n:.3f}")
    
    print(f"\n[Final Luminance Grid Mechanics]")
    print(f"ASR: {asr_t:.1f}% | Metric LPIPS: {lpips_t:.3f}")

    print(f"\n[Final Adversarial Patch Mechanics]")
    print(f"ASR: {asr_p:.1f}% | Metric LPIPS: {lpips_p:.3f}")
    
    print("\n" + "="*50)
    print("LaTeX Table 9 Synthesis Ready (Copy / Paste Below):")
    print("-" * 50)
    print(f"Structure & ASR & LPIPS \\\\")
    print(f"\\midrule")
    print(f"Adversarial Patch & {asr_p:.1f}\\% & {lpips_p:.3f} \\\\")
    print(f"Narrowband Feature Mimicry & {asr_n:.1f}\\% & {lpips_n:.3f} \\\\")
    print(f"Luminance Grid & \\mathbf{{{asr_t:.1f}\\%}} & {lpips_t:.3f} \\\\")
    print("="*50 + "\n")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out = {
        "Narrowband_ASR": asr_n, 
        "Narrowband_LPIPS": lpips_n,
        "LuminanceGrid_ASR": asr_t,
        "LuminanceGrid_LPIPS": lpips_t,
        "AdvPatch_ASR": asr_p,
        "AdvPatch_LPIPS": lpips_p
    }
    with open(os.path.join(results_dir, 'structural_mechanisms.json'), 'w') as f:
        json.dump(out, f, indent=4)

if __name__ == "__main__":
    main()
