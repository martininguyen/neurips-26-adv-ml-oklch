import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import numpy as np
from diffusers import AutoencoderKL

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def sd35_autoencoder_purify(img_tensor, vae):
    with torch.no_grad():
        tensor_bf = img_tensor.to(torch.float16)
        scaled = tensor_bf * 2.0 - 1.0
        encoded = vae.encode(scaled).latent_dist.sample()
        output = vae.decode(encoded).sample
        return ((output + 1.0) / 2.0).clamp(0, 1).to(torch.float32)

def evaluate_batch_accuracy(model, normalize_fn, img_tensor, labels):
    with torch.no_grad():
        logits = model(normalize_fn(img_tensor))
        preds = logits.argmax(dim=1)
        correct_mask = (preds == labels)
    return correct_mask, preds

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "chromacrypt_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    num_test = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    patch_size = config["purification"]["patch_size"]
    chromic_amp = config["ThreatMappings"]["eps_structural"]
    vae_id = config["purification"]["sd35_vae_model_id"]
    
    print(f"Loading SOTA Convolutional Autoencoder: {vae_id} (Float16 Constraint)...")
    try:
        core_utils.load_env()
        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        vae = AutoencoderKL.from_pretrained(
            vae_id, 
            subfolder="vae",
            torch_dtype=torch.float16,
            token=hf_token
        ).to(DEVICE)
        vae.eval()
    except Exception as e:
        try:
            print(f"HuggingFace Remote 503 Encountered. Attempting Native Local Cache Resolution for {vae_id}...")
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=torch.float16, token=hf_token, local_files_only=True).to(DEVICE)
            vae.eval()
        except:
            print(f"CRITICAL VAE LOADING ERROR: {e}\nEnsure HuggingFace token exists indicating SD3.5 CLA authorization.")
            sys.exit(1)
        
    print(f"Loading ResNet50 Classifier Baseline...")
    model = core_utils.load_victim_model().to(DEVICE)
    model.eval()
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    normalize_fn = lambda x: (x - mean) / std

    total_images = 0
    patch_pre = 0; patch_post = 0; patch_rescued = 0; patch_hallucinated = 0
    narrow_pre = 0; narrow_post = 0; narrow_rescued = 0; narrow_hallucinated = 0
    grid_pre = 0; grid_post = 0; grid_rescued = 0; grid_hallucinated = 0
    grid_low_pre = 0; grid_low_post = 0; grid_low_rescued = 0; grid_low_hallucinated = 0

    print("Executing iterative memory-safe SD3.5 evaluation loop...")
    for offset in range(0, num_test, batch_size):
        curr_batch_size = min(batch_size, num_test - offset)
        c_images, labels, _, _ = core_utils.load_imagenet_val_batch(curr_batch_size, offset=offset)
        
        c_images = c_images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        clean_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, c_images, labels)
        valid_indices = torch.where(clean_correct_mask)[0]
        if len(valid_indices) == 0:
            continue
            
        total_images += len(valid_indices)
        
        patch_atk = cc.AdvPatch(model=model, patch_size=patch_size)
        narrow_atk = cc.NarrowbandMimicry(eps=chromic_amp)
        grid_atk = cc.TopologicalAttractor(eps=chromic_amp)
        grid_low_atk = cc.TopologicalAttractor(eps=0.05)
        
        patch_adv = patch_atk(c_images, labels=labels)
        narrow_adv = narrow_atk(c_images, color_ops)
        grid_adv = grid_atk(c_images, color_ops)
        grid_low_adv = grid_low_atk(c_images, color_ops)

        patch_pre_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, patch_adv, labels)
        narrow_pre_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, narrow_adv, labels)
        grid_pre_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, grid_adv, labels)
        grid_low_pre_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, grid_low_adv, labels)
        
        patch_pre += (1 - patch_pre_correct_mask[valid_indices].int()).sum().item()
        narrow_pre += (1 - narrow_pre_correct_mask[valid_indices].int()).sum().item()
        grid_pre += (1 - grid_pre_correct_mask[valid_indices].int()).sum().item()
        grid_low_pre += (1 - grid_low_pre_correct_mask[valid_indices].int()).sum().item()
        
        patch_purified = sd35_autoencoder_purify(patch_adv, vae)
        narrow_purified = sd35_autoencoder_purify(narrow_adv, vae)
        grid_purified = sd35_autoencoder_purify(grid_adv, vae)
        grid_low_purified = sd35_autoencoder_purify(grid_low_adv, vae)

        patch_post_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, patch_purified, labels)
        narrow_post_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, narrow_purified, labels)
        grid_post_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, grid_purified, labels)
        grid_low_post_correct_mask, _ = evaluate_batch_accuracy(model, normalize_fn, grid_low_purified, labels)

        patch_post += (1 - patch_post_correct_mask[valid_indices].int()).sum().item()
        narrow_post += (1 - narrow_post_correct_mask[valid_indices].int()).sum().item()
        grid_post += (1 - grid_post_correct_mask[valid_indices].int()).sum().item()
        grid_low_post += (1 - grid_low_post_correct_mask[valid_indices].int()).sum().item()

        """
        [Logic Block]
        Operation: Latent Recovery Topography (Rescue vs Hallucination Tracker)
        Algebra:
           1. Rescued = (~Correct_{Pre}) & (Correct_{Post})
              -> Mathematical proof that diffusion bottleneck normalized adversarial geometry constraints natively filtering topological noise.
           2. Hallucinated = (Correct_{Pre}) & (~Correct_{Post})
              -> Proves that SD 3.5 spatial restructuring over-abstracted valid parameters destroying classification boundary confidence independently of attacks.
        Purpose: Ensures performance matrices do not falsely obfuscate structural weaknesses behind native VAE degradation failure metrics.
        """
        patch_rescued += ((~patch_pre_correct_mask[valid_indices]) & patch_post_correct_mask[valid_indices]).sum().item()
        patch_hallucinated += (patch_pre_correct_mask[valid_indices] & (~patch_post_correct_mask[valid_indices])).sum().item()

        narrow_rescued += ((~narrow_pre_correct_mask[valid_indices]) & narrow_post_correct_mask[valid_indices]).sum().item()
        narrow_hallucinated += (narrow_pre_correct_mask[valid_indices] & (~narrow_post_correct_mask[valid_indices])).sum().item()

        grid_rescued += ((~grid_pre_correct_mask[valid_indices]) & grid_post_correct_mask[valid_indices]).sum().item()
        grid_hallucinated += (grid_pre_correct_mask[valid_indices] & (~grid_post_correct_mask[valid_indices])).sum().item()

        grid_low_rescued += ((~grid_low_pre_correct_mask[valid_indices]) & grid_low_post_correct_mask[valid_indices]).sum().item()
        grid_low_hallucinated += (grid_low_pre_correct_mask[valid_indices] & (~grid_low_post_correct_mask[valid_indices])).sum().item()

        print(f"  -> Processed {total_images} Extrapolations seamlessly")

    def calc(val): return (val/total_images)*100 if total_images > 0 else 0

    print("\n" + "="*50)
    print("LaTeX Table 13 Purification Breakdown Synthesis Ready:")
    print("-" * 50)
    print(f"Physical Realization & Architecture Amplitude & Original ASR & Post-Purification (SD 3.5) ASR & Net Rescued / Hallucinated \\\\")
    print(f"\\midrule")
    print(f"Adversarial Patch & Box $= {patch_size}\\times{patch_size}$ & {calc(patch_pre):.1f}\\% & \\mathbf{{{calc(patch_post):.1f}\\%}} & (+{patch_rescued} / -{patch_hallucinated}) \\\\")
    print(f"Narrowband Feature Mimicry & Bandwidth $= 2.0$ & {calc(narrow_pre):.1f}\\% & \\mathbf{{{calc(narrow_post):.1f}\\%}} & (+{narrow_rescued} / -{narrow_hallucinated}) \\\\")
    print(f"Luminance Grid & Amp $= {chromic_amp}$ & {calc(grid_pre):.1f}\\% & \\mathbf{{{calc(grid_post):.1f}\\%}} & (+{grid_rescued} / -{grid_hallucinated}) \\\\")
    print(f"Luminance Grid & Amp $= 0.05$ & {calc(grid_low_pre):.1f}\\% & \\mathbf{{{calc(grid_low_post):.1f}\\%}} & (+{grid_low_rescued} / -{grid_low_hallucinated}) \\\\")
    print("="*50 + "\n")
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "Adversarial Patch": {"Pre_ASR": calc(patch_pre)/100, "Post_ASR": calc(patch_post)/100, "Rescued": patch_rescued, "Hallucinated": patch_hallucinated},
        "Narrowband Feature Mimicry": {"Pre_ASR": calc(narrow_pre)/100, "Post_ASR": calc(narrow_post)/100, "Rescued": narrow_rescued, "Hallucinated": narrow_hallucinated},
        "Luminance Grid": {"Pre_ASR": calc(grid_pre)/100, "Post_ASR": calc(grid_post)/100, "Rescued": grid_rescued, "Hallucinated": grid_hallucinated},
        "Luminance Grid (A=0.05)": {"Pre_ASR": calc(grid_low_pre)/100, "Post_ASR": calc(grid_low_post)/100, "Rescued": grid_low_rescued, "Hallucinated": grid_low_hallucinated}
    }
    with open(os.path.join(out_dir, "table13_sd35_purification.json"), "w") as f: json.dump({"Total_Evaluated": total_images, "results": out}, f, indent=4)

if __name__ == "__main__":
    main()
