import os
import json
import torch
import numpy as np
import lpips
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils
from diffusers import StableDiffusion3Img2ImgPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_fft_peak_magnitude(image_tensor, color_ops):
    b, c, h, w = image_tensor.shape
    oklch = color_ops.rgb_to_oklch(image_tensor)
    L_channel = oklch[:, 0, :, :]
    fft_res = torch.fft.rfft2(L_channel)
    magnitude = torch.sqrt(fft_res.real**2 + fft_res.imag**2)
    freq_y = int(h / 16.0)
    freq_x = int(w / 16.0)
    
    # 3x3 local window to catch VAE spectral smearing
    y_start, y_end = max(0, freq_y - 1), freq_y + 2
    x_start, x_end = max(0, freq_x - 1), freq_x + 2
    peak_magnitudes = magnitude[:, y_start:y_end, x_start:x_end].amax(dim=(1, 2))
    return peak_magnitudes

def main():
    print("Loading SD3.5 for Sweep...")
    core_utils.load_env()
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    
    pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=torch.float16, 
        token=hf_token, 
        safety_checker=None
    ).to(DEVICE)
    pipeline.set_progress_bar_config(disable=False)
    
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    
    amplitudes = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    num_images = 50 # Small subset for speed
    batch_size = 10
    
    results = {}
    
    for A in amplitudes:
        print(f"\n--- Testing Amplitude A={A} ---")
        grid_atk = cc.TopologicalAttractor(eps=A, channel="LC")
        
        lpips_scores = []
        post_lpips_scores = [] # Tracking Latent Collapse!
        survival_ratios = []
        
        for offset in range(0, num_images, batch_size):
            curr_batch_size = min(batch_size, num_images - offset)
            print(f"  -> Processing batch {offset // batch_size + 1}/{(num_images + batch_size - 1) // batch_size}...")
            images, labels, _, _ = core_utils.load_imagenet_val_batch(curr_batch_size, offset=offset)
            images = images.to(DEVICE)
            
            # FIX: Upsample to 512x512 to prevent SD3.5 RoPE OOD Collapse
            images = torch.nn.functional.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
            
            # 1. Attack
            adv_images = grid_atk(images, color_ops)
            
            # 2. LPIPS (Pre-Purification)
            clean_norm = images * 2.0 - 1.0
            adv_norm = adv_images * 2.0 - 1.0
            with torch.no_grad():
                batch_lpips = lpips_fn(clean_norm, adv_norm).squeeze().cpu().numpy()
            if batch_lpips.ndim == 0: batch_lpips = [batch_lpips.item()]
            else: batch_lpips = batch_lpips.tolist()
            lpips_scores.extend(batch_lpips)
            
            # 3. Purify
            pre_fft = get_fft_peak_magnitude(adv_images, color_ops)
            
            # Lock the SDE generator seed so your sweep data is 100% reproducible
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            
            purified_tensors = []
            for i in range(curr_batch_size):
                img_pil = core_utils.tensor_to_pil(adv_images[i])
                with torch.no_grad():
                    pur_img = pipeline(
                        prompt="", 
                        image=img_pil, 
                        strength=0.35, 
                        guidance_scale=0.0,
                        generator=gen
                    ).images[0]
                purified_tensors.append(core_utils.pil_to_tensor(pur_img).to(DEVICE))
            purified = torch.stack(purified_tensors)
            
            # 4. Measure Grid Survival
            post_fft = get_fft_peak_magnitude(purified, color_ops)
            ratio = (post_fft / (pre_fft + 1e-8)).cpu().tolist()
            survival_ratios.extend(ratio)
            
            # 5. Measure Latent Collapse (Post-Purification LPIPS)
            purified_norm = purified * 2.0 - 1.0
            with torch.no_grad():
                post_pur_lpips = lpips_fn(clean_norm, purified_norm).squeeze().cpu().numpy()
            if post_pur_lpips.ndim == 0: post_pur_lpips = [post_pur_lpips.item()]
            else: post_pur_lpips = post_pur_lpips.tolist()
            post_lpips_scores.extend(post_pur_lpips)
            
        mean_lpips = np.mean(lpips_scores)
        mean_post_lpips = np.mean(post_lpips_scores)
        mean_survival = np.mean(survival_ratios) * 100
        
        print(f"Results for A={A}: Pre-LPIPS={mean_lpips:.4f} | Post-LPIPS (Collapse Metric)={mean_post_lpips:.4f} | Survival={mean_survival:.2f}%")
        results[str(A)] = {
            "pre_lpips": mean_lpips, 
            "post_lpips": mean_post_lpips, 
            "survival_pct": mean_survival
        }
        
    print("\nSweep Complete:", results)

if __name__ == "__main__":
    main()
