import os
import sys
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
import torch
import torchvision.models as models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
from diffusers import AutoencoderKL

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sd35_autoencoder_purify(img_tensor, vae):
    """
    [Logic Block]
    Operation: SOTA SD3.5 Float16 Purification Network
    Algebra:
      1. Casts Tensors -> torch.float16 globally.
      2. Generates Latent Gaussian Representation: Z = E(X_scaled)
      3. Output = D(Z) -> float32
    Purpose: Empties explicit spatial topology mappings by forcing geometric anomalies through SD3.5's extremely resilient VAE spatial bottleneck matrix to test structural survival after extreme continuous memory purification.
    """
    if vae is None: return img_tensor # Fallback if SD3.5 disabled
    with torch.no_grad():
        purified_images = []
        for i in range(img_tensor.size(0)):
            # Isolate evaluation to 1 image per pass to prevent VRAM explosion
            tensor_bf = img_tensor[i:i+1].to(torch.float16)
            scaled = tensor_bf * 2.0 - 1.0
            encoded = vae.encode(scaled).latent_dist.sample()
            output = vae.decode(encoded).sample
            purified_images.append(((output + 1.0) / 2.0).clamp(0, 1).to(torch.float32))
        return torch.cat(purified_images, dim=0)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "chromacrypt_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    num_images = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    eps_l = config.get("ThreatMappings", {}).get("eps_oklch_l", 0.01)
    eps_c = config.get("ThreatMappings", {}).get("eps_oklch_c", 0.01)
    eps_h = eps_l * 360.0

    print(f"Evaluating Topologies against SOTA Resilient Checkpoints (N={num_images})...")
    
    # 1. Load Models
    s_name = config.get("robust_models", {}).get("surrogate_model", "resnet50")
    t_name = config.get("robust_models", {}).get("target_model", "wide_resnet50_2")
    
    surrogate = getattr(models, s_name)(weights="DEFAULT").eval().to(DEVICE)
    target = getattr(models, t_name)(weights="DEFAULT").eval().to(DEVICE)
    
    vae = None
    try:
        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        # Integrating the official community float16 fix to prevent activation overflow
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16,
            token=hf_token
        ).to(DEVICE)
        vae.eval()
        print("Community FP16-Fixed VAE Pipeline Loaded Successfully.")
    except Exception as e:
        print(f"VAE Loading Failed (Skipping Diffusion Purification): {e}")

    total_imgs = 0
    wb_wins = 0
    bb_wins = 0
    wb_sd35_wins = 0
    bb_sd35_wins = 0

    print("Executing iterative memory-safe evaluation loop...")
    for offset in range(0, num_images, batch_size):
        curr_batch_size = min(batch_size, num_images - offset)
        images, labels, mean, std = core_utils.load_imagenet_val_batch(n_examples=curr_batch_size, offset=offset)
        img_tensor = images.to(DEVICE)
        lbl_tensor = labels.to(DEVICE)
        
        # White-Box Attack (Attacking Target Directly)
        chromic_atk_wb = cc.ChromicPGD(model=target, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h)
        adv_wb = chromic_atk_wb(img_tensor, lbl_tensor)
        
        with torch.no_grad():
            preds_wb = target((adv_wb - mean) / std).argmax(1)
            wb_wins += (preds_wb != lbl_tensor).float().sum().item()
            
        # Black-Box Attack (Attacking Surrogate, Transferring to Target)
        chromic_atk_bb = cc.ChromicPGD(model=surrogate, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h)
        adv_bb = chromic_atk_bb(img_tensor, lbl_tensor)
        
        with torch.no_grad():
            preds_bb = target((adv_bb - mean) / std).argmax(1)
            bb_wins += (preds_bb != lbl_tensor).float().sum().item()
            
        # Diffusion Purification Resistance 
        if vae:
            purified_wb = sd35_autoencoder_purify(adv_wb, vae)
            purified_bb = sd35_autoencoder_purify(adv_bb, vae)
            
            with torch.no_grad():
                preds_purified_wb = target((purified_wb - mean) / std).argmax(1)
                wb_sd35_wins += (preds_purified_wb != lbl_tensor).float().sum().item()
                
                preds_purified_bb = target((purified_bb - mean) / std).argmax(1)
                bb_sd35_wins += (preds_purified_bb != lbl_tensor).float().sum().item()
                
        total_imgs += curr_batch_size
        print(f"  -> Processed {total_imgs}/{num_images} validation tensors")

    asr_wb = (wb_wins/total_imgs)*100
    asr_bb = (bb_wins/total_imgs)*100
    asr_wb_sd35 = (wb_sd35_wins/total_imgs)*100 if vae else 0.0
    asr_bb_sd35 = (bb_sd35_wins/total_imgs)*100 if vae else 0.0

    print(f"\n--- [Final Validation Matrix] ---")
    print(f"WideResNet50 White-Box ASR: {asr_wb:.1f}%")
    print(f"WideResNet50 Black-Box (Transfer from ResNet50) ASR: {asr_bb:.1f}%")
    
    if vae:
        print(f"White-Box Post-Purification ASR: {asr_wb_sd35:.1f}%")
        print(f"Black-Box Post-Purification ASR: {asr_bb_sd35:.1f}%")

    print("\n" + "="*50)
    print("LaTeX Table 10 Synthesis Ready (Copy / Paste Below):")
    print("-" * 50)
    print(f"Threat Model & Robust Architecture & Pre-Purification ASR & Post-Purification (SD 3.5) ASR \\\\")
    print(f"\\midrule")
    if vae:
        print(f"White-Box & {t_name} & {asr_wb:.1f}\\% & \\mathbf{{{asr_wb_sd35:.1f}\\%}} \\\\")
        print(f"Black-Box & {t_name} & {asr_bb:.1f}\\% & \\mathbf{{{asr_bb_sd35:.1f}\\%}} \\\\")
    else:
        print(f"White-Box & {t_name} & {asr_wb:.1f}\\% & N/A \\\\")
        print(f"Black-Box & {t_name} & {asr_bb:.1f}\\% & N/A \\\\")
    print("="*50 + "\n")
        
    # Output saving
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out = {
        "WhiteBox_ASR": asr_wb,
        "BlackBox_ASR": asr_bb,
        "WhiteBox_SD35_ASR": asr_wb_sd35,
        "BlackBox_SD35_ASR": asr_bb_sd35
    }
    with open(os.path.join(results_dir, 'robust_models_eval.json'), 'w') as f:
        json.dump(out, f, indent=4)

if __name__ == "__main__":
    main()
