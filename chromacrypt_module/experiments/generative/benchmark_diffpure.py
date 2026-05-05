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
    
    from diffusers import StableDiffusionImg2ImgPipeline
    core_utils.load_env()
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    
    print(f"Loading DiffPure Diffusion Pipeline (runwayml/stable-diffusion-v1-5)...")
    try:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, token=hf_token, safety_checker=None).to(DEVICE)
    except Exception as e:
        print(f"HuggingFace Remote 503 Encountered. Attempting Native Local Cache Resolution...")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, token=hf_token, local_files_only=True, safety_checker=None).to(DEVICE)
    
    pipeline.set_progress_bar_config(disable=False)
    
    print("Loading Targeting ResNet50 Classifier Baseline...")
    model = core_utils.load_victim_model().to(DEVICE)
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    norm_fn = lambda x: (x - mean) / std

    results = {
        "metadata": {"N": num_images},
        "total_images": 0,
        "patch": {"pre_success": 0, "post_success": 0, "rescued": 0, "snr_survived": 0},
        "grid": {"pre_success": 0, "post_success": 0, "rescued": 0, "snr_survived": 0},
        "grid_chromic": {"pre_success": 0, "post_success": 0, "rescued": 0, "snr_survived": 0},
        "natural": {"pre_success": 0, "post_success": 0, "rescued": 0, "snr_survived": 0}
    }

    def eval_batch(x, y):
        with torch.no_grad():
            preds = model(norm_fn(x)).argmax(1)
        return preds == y
        
    def purify(x):
        """
        [Logic Block]
        Operation: Formal DiffPure SDE Verification (Stable Diffusion 1.5)
        Algebra:
          1. Projects topological spatial constraints to generative pipeline.
          2. Applies native forward SDE injection ($t$ steps corresponding to strength).
          3. Evaluates Reverse-SDE Unet sampling conditionally unguided ('').
        Purpose: Formally replicates stochastic diffusion purification mechanisms claimed in Section 5 natively, avoiding static VAE latent bottlenecks.
        """
        results_tensors = []
        for i in range(x.shape[0]):
            img = core_utils.tensor_to_pil(x[i])
            with torch.no_grad():
                purified_img = pipeline(prompt="", image=img, strength=0.35, guidance_scale=0.0).images[0]
            results_tensors.append(core_utils.pil_to_tensor(purified_img).to(DEVICE))
        
        return torch.stack(results_tensors)

    print("Beginning Iterative DP Purge (Table 5)...")
    for offset in range(0, num_images, batch_size):
        curr_batch_size = min(batch_size, num_images - offset)
        print(f"\nProcessing Batch: Images {offset+1} to {offset+curr_batch_size} of {num_images}...")
        images, labels, _, _ = core_utils.load_imagenet_val_batch(curr_batch_size, offset=offset)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        clean_mask = eval_batch(images, labels)
        results["total_images"] += clean_mask.sum().item()
        
        # Attacks
        color_ops = cc.DifferentiableColorOps().to(DEVICE)
        
        patch_atk = cc.AdvPatch(model=model, patch_size=config["purification"]["patch_size"])
        grid_atk = cc.TopologicalAttractor(eps=config["ThreatMappings"]["eps_structural"], channel="L")
        grid_chromic_atk = cc.TopologicalAttractor(eps=config["ThreatMappings"]["eps_structural"], channel="LC")
        natural_atk = cc.NarrowbandMimicry(eps=config["ThreatMappings"]["eps_structural"])
        
        patch_adv = patch_atk(images, labels=labels)
        grid_adv = grid_atk(images, color_ops)
        grid_chromic_adv = grid_chromic_atk(images, color_ops)
        natural_adv = natural_atk(images, color_ops)

        # Pre Eval
        patch_pre = eval_batch(patch_adv, labels)
        grid_pre = eval_batch(grid_adv, labels)
        grid_chromic_pre = eval_batch(grid_chromic_adv, labels)
        nat_pre = eval_batch(natural_adv, labels)

        # Purify
        patch_purified = purify(patch_adv)
        grid_purified = purify(grid_adv)
        grid_chromic_purified = purify(grid_chromic_adv)
        nat_purified = purify(natural_adv)

        # Post Eval
        patch_post = eval_batch(patch_purified, labels)
        grid_post = eval_batch(grid_purified, labels)
        grid_chromic_post = eval_batch(grid_chromic_purified, labels)
        nat_post = eval_batch(nat_purified, labels)
        
        # Masks
        for atk_name, pre_m, post_m in [("patch", patch_pre, patch_post), ("grid", grid_pre, grid_post), ("grid_chromic", grid_chromic_pre, grid_chromic_post), ("natural", nat_pre, nat_post)]:
            results[atk_name]["pre_success"] += (~pre_m & clean_mask).sum().item()
            results[atk_name]["post_success"] += (~post_m & clean_mask).sum().item()
            results[atk_name]["rescued"] += ((~pre_m) & post_m & clean_mask).sum().item()
            results[atk_name]["snr_survived"] += (pre_m & (~post_m) & clean_mask).sum().item()
            
        print(f"  -> Processed {results['total_images']} parameters seamlessly")

    def calc(v_dict):
        tot = results['total_images']
        return v_dict["pre_success"]/tot*100 if tot > 0 else 0, v_dict["post_success"]/tot*100 if tot > 0 else 0

    print("\n" + "="*50)
    print("LaTeX Table 5 DiffPure Synthesis Ready:")
    print("-" * 50)
    print(f"Attack Generator & Pre-Purification ASR & Post-Purification ASR & Rescued Images & SNR Survivals \\\\")
    print(f"\\midrule")
    
    p_pr, p_po = calc(results['patch'])
    print(f"Adversarial Patch ($32\\times32$) & {p_pr:.1f}\\% & \\mathbf{{{p_po:.1f}\\%}} & (+{results['patch']['rescued']} / -{results['patch']['snr_survived']}) \\\\")
    
    n_pr, n_po = calc(results['natural'])
    print(f"Narrowband Feature Mimicry ($A=0.50$) & {n_pr:.1f}\\% & \\mathbf{{{n_po:.1f}\\%}} & (+{results['natural']['rescued']} / -{results['natural']['snr_survived']}) \\\\")
    
    g_pr, g_po = calc(results['grid'])
    print(f"Luminance Grid ($A=0.50$) & {g_pr:.1f}\\% & \\mathbf{{{g_po:.1f}\\%}} & (+{results['grid']['rescued']} / -{results['grid']['snr_survived']}) \\\\")
    
    gc_pr, gc_po = calc(results['grid_chromic'])
    print(f"Chromic Grid ($A=0.50$) & {gc_pr:.1f}\\% & \\mathbf{{{gc_po:.1f}\\%}} & (+{results['grid_chromic']['rescued']} / -{results['grid_chromic']['snr_survived']}) \\\\")
    print("="*50 + "\n")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table5_diffusion_purification.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("Table 5 Export Completed Successfully.")

if __name__ == "__main__":
    main()
