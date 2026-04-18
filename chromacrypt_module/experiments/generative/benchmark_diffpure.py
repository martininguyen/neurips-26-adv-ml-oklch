import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import time
import numpy as np

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromacrypt_config.json"), "r") as f:
        config = json.load(f)
        
    num_images = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    
    from diffusers import AutoencoderKL
    vae_id = config["purification"]["sd15_vae_model_id"]
    print(f"Loading DiffPure VAE Base Purification Sequence: {vae_id}...")
    core_utils.load_env()
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    try:
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16, token=hf_token).to(DEVICE)
    except Exception as e:
        print(f"HuggingFace Remote 503 Encountered. Attempting Native Local Cache Resolution for {vae_id}...")
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16, token=hf_token, local_files_only=True).to(DEVICE)
    vae.eval()
    
    print("Loading Targeting ResNet50 Classifier Baseline...")
    model = core_utils.load_victim_model().to(DEVICE)
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    norm_fn = lambda x: (x - mean) / std

    results = {
        "metadata": {"N": num_images},
        "total_images": 0,
        "patch": {"pre_success": 0, "post_success": 0, "rescued": 0, "hallucinated": 0},
        "grid": {"pre_success": 0, "post_success": 0, "rescued": 0, "hallucinated": 0},
        "natural": {"pre_success": 0, "post_success": 0, "rescued": 0, "hallucinated": 0}
    }

    def eval_batch(x, y):
        with torch.no_grad():
            preds = model(norm_fn(x)).argmax(1)
        return preds == y
        
    def purify(x):
        """
        [Logic Block]
        Operation: Autoencoder Purification Matrix (DiffPure Simulation)
        Algebra:
          1. Scales pixel tensor: V_{in} = X_{fp16} \times 2 - 1
          2. Encodes to bounded structural distribution: Z = E(V_{in})
          3. S_Z = Sample(Z)
          4. Decodes topology: P_{fp16} = D(S_Z)
          5. Recalibrates numeric bounds natively: Output = ((P + 1) / 2) \to fp32
        Purpose: Emulates baseline generative structural defense by driving image inputs through a low-dimensional topological compression bottleneck, systematically neutralizing mathematically perfect unstructured adversarial norms.
        """
        with torch.no_grad():
            vae_in = x.half() * 2.0 - 1.0
            latents = vae.encode(vae_in).latent_dist.sample()
            purified = vae.decode(latents).sample
            return ((purified + 1.0) / 2.0).clamp(0, 1).float()

    print("Beginning Iterative DP Purge (Table 11)...")
    for offset in range(0, num_images, batch_size):
        curr_batch_size = min(batch_size, num_images - offset)
        images, labels, _, _ = core_utils.load_imagenet_val_batch(curr_batch_size, offset=offset)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        clean_mask = eval_batch(images, labels)
        results["total_images"] += clean_mask.sum().item()
        
        # Attacks
        color_ops = cc.DifferentiableColorOps().to(DEVICE)
        
        patch_atk = cc.AdvPatch(model=model, patch_size=config["purification"]["patch_size"])
        grid_atk = cc.TopologicalAttractor(eps=config["ThreatMappings"]["eps_structural"])
        natural_atk = cc.NarrowbandMimicry(eps=config["ThreatMappings"]["eps_structural"])
        
        patch_adv = patch_atk(images, labels=labels)
        grid_adv = grid_atk(images, color_ops)
        natural_adv = natural_atk(images, color_ops)

        # Pre Eval
        patch_pre = eval_batch(patch_adv, labels)
        grid_pre = eval_batch(grid_adv, labels)
        nat_pre = eval_batch(natural_adv, labels)

        # Purify
        patch_purified = purify(patch_adv)
        grid_purified = purify(grid_adv)
        nat_purified = purify(natural_adv)

        # Post Eval
        patch_post = eval_batch(patch_purified, labels)
        grid_post = eval_batch(grid_purified, labels)
        nat_post = eval_batch(nat_purified, labels)
        
        # Masks
        for atk_name, pre_m, post_m in [("patch", patch_pre, patch_post), ("grid", grid_pre, grid_post), ("natural", nat_pre, nat_post)]:
            results[atk_name]["pre_success"] += (~pre_m & clean_mask).sum().item()
            results[atk_name]["post_success"] += (~post_m & clean_mask).sum().item()
            results[atk_name]["rescued"] += ((~pre_m) & post_m & clean_mask).sum().item()
            results[atk_name]["hallucinated"] += (pre_m & (~post_m) & clean_mask).sum().item()
            
        print(f"  -> Processed {results['total_images']} parameters seamlessly")

    def calc(v_dict):
        tot = results['total_images']
        return v_dict["pre_success"]/tot*100 if tot > 0 else 0, v_dict["post_success"]/tot*100 if tot > 0 else 0

    print("\n" + "="*50)
    print("LaTeX Table 11 DiffPure Synthesis Ready:")
    print("-" * 50)
    print(f"Attack Generator & Pre-Purification ASR & Post-Purification ASR & Rescued Images & Hallucinations \\\\")
    print(f"\\midrule")
    
    p_pr, p_po = calc(results['patch'])
    print(f"Adversarial Patch ($32\\times32$) & {p_pr:.1f}\\% & \\mathbf{{{p_po:.1f}\\%}} & (+{results['patch']['rescued']} / -{results['patch']['hallucinated']}) \\\\")
    
    n_pr, n_po = calc(results['natural'])
    print(f"Narrowband Feature Mimicry ($A=0.50$) & {n_pr:.1f}\\% & \\mathbf{{{n_po:.1f}\\%}} & (+{results['natural']['rescued']} / -{results['natural']['hallucinated']}) \\\\")
    
    g_pr, g_po = calc(results['grid'])
    print(f"Luminance Grid ($A=0.50$) & {g_pr:.1f}\\% & \\mathbf{{{g_po:.1f}\\%}} & (+{results['grid']['rescued']} / -{results['grid']['hallucinated']}) \\\\")
    print("="*50 + "\n")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table11_diffusion_purification.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("Table 11 Export Completed Successfully.")

if __name__ == "__main__":
    main()
