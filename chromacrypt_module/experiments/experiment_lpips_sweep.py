import sys
import os
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromacrypt_module.utils import load_victim_model, load_imagenet_val_batch, DEVICE
from chromacrypt_module.attacks import NarrowbandMimicry, TopologicalAttractor
from chromacrypt_module.color_ops import DifferentiableColorOps
import lpips
from diffusers import AutoencoderKL

def run_lpips_sweep():
    print("Initializing LPIPS Sweep Experiment...")
    device = DEVICE
    
    # Load Evaluators
    target_model = load_victim_model()
    color_ops = DifferentiableColorOps().to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # Load Diffusion Purifier (SD 3.5 VAE as proxy for diffusion bottleneck)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae").to(device)
    vae.eval()
    
    # Load Data
    batch, labels, mean, std = load_imagenet_val_batch(n_examples=20, offset=0)
    
    # Epsilon intervals requested: 0.05, 0.2, 0.4, 0.6, 0.8
    eps_values = [0.05, 0.2, 0.4, 0.6, 0.8]
    
    results = {
        "Chromic Grid (LC)": [],
        "Luminance Grid (L)": [],
        "Narrowband Mimicry (LC)": []
    }
    
    print(f"Executing sweep over epsilons: {eps_values}")
    
    for eps in eps_values:
        print(f"\n--- Evaluating eps = {eps} ---")
        
        attacks = {
            "Chromic Grid (LC)": TopologicalAttractor(eps=eps, channel="LC"),
            "Luminance Grid (L)": TopologicalAttractor(eps=eps, channel="L"),
            "Narrowband Mimicry (LC)": NarrowbandMimicry(eps=eps, channel="LC")
        }
        
        for name, attack in attacks.items():
            # 1. Generate Perturbation
            adv_images = attack(batch, color_ops)
            
            # 2. Calculate LPIPS (Inputs must be in [-1, 1])
            lpips_dist = loss_fn_vgg(batch * 2 - 1, adv_images * 2 - 1).mean().item()
            
            # 3. Base ASR
            norm_adv = (adv_images - mean) / std
            with torch.no_grad():
                preds = target_model(norm_adv).argmax(1)
                acc = (preds == labels).float().mean().item()
                asr = 1.0 - acc
                
            # 4. Post-Purification ASR (Surviving the VAE bottleneck)
            with torch.no_grad():
                latents = vae.encode(adv_images * 2 - 1).latent_dist.sample()
                purified_images = vae.decode(latents).sample
                purified_images = (purified_images / 2 + 0.5).clamp(0, 1)
                
                norm_purified = (purified_images - mean) / std
                purified_preds = target_model(norm_purified).argmax(1)
                purified_acc = (purified_preds == labels).float().mean().item()
                post_diffusion_asr = 1.0 - purified_acc
                
            results[name].append({
                "eps": eps,
                "lpips": lpips_dist,
                "asr": asr,
                "post_diffusion_asr": post_diffusion_asr
            })
            
            print(f"[{name}] LPIPS: {lpips_dist:.4f} | Pre-Purification ASR: {asr*100:.2f}% | Post-Purification ASR: {post_diffusion_asr*100:.2f}%")
            
    print("\n--- Sweep Complete ---")
    return results

if __name__ == "__main__":
    run_lpips_sweep()
