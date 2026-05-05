import torch
import torch.nn as nn
import os
import sys
import json
import numpy as np
import gc

# Inject canonical chromacrypt pathing
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_fft_peak_magnitude(image_tensor, color_ops):
    """
    Extracts the magnitude of the spatial frequency corresponding to the Chromic Grid (16-pixel period).
    """
    b, c, h, w = image_tensor.shape
    
    # Convert to OKLCH to isolate the structure in the Luminance channel
    oklch = color_ops.rgb_to_oklch(image_tensor)
    L_channel = oklch[:, 0, :, :] # [B, H, W]
    
    # Compute 2D FFT
    fft_res = torch.fft.rfft2(L_channel)
    magnitude = torch.sqrt(fft_res.real**2 + fft_res.imag**2)
    
    freq_y = int(h / 16.0)
    freq_x = int(w / 16.0)
    
    peak_magnitudes = magnitude[:, freq_y, freq_x]
    
    return peak_magnitudes

def get_narrowband_magnitude(image_tensor, color_ops, freq_mult=1.0, bw=2.0):
    """
    Extracts the magnitude of the spatial frequencies corresponding to 
    the Narrowband Feature Mimicry annulus.
    """
    b, c, h, w = image_tensor.shape
    device = image_tensor.device
    
    # 1. Convert to OKLCH and isolate the Luminance channel
    oklch = color_ops.rgb_to_oklch(image_tensor)
    L_channel = oklch[:, 0, :, :] 
    
    # 2. Compute full 2D FFT 
    # (Using fft2 instead of rfft2 to exactly match the meshgrid used in generation)
    fft_res = torch.fft.fft2(L_channel)
    magnitude = torch.sqrt(fft_res.real**2 + fft_res.imag**2)
    
    # 3. Recreate the Gaussian spectral filter G(wx, wy)
    base_freq = 224.0 / 16.0 
    target_k = base_freq * freq_mult
    sigma = bw * freq_mult
    
    fx = torch.fft.fftfreq(w, d=1.0).to(device) * w
    fy = torch.fft.fftfreq(h, d=1.0).to(device) * h
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    
    rad_dist = torch.sqrt(FX**2 + FY**2)
    
    # 4. Apply the exact same target band mask used during generation
    mask = torch.exp(-((rad_dist - target_k)**2) / (2 * sigma**2))
    mask = mask.unsqueeze(0).expand(b, h, w)
    
    # 5. Extract the weighted amplitude across the target annulus
    band_magnitude = torch.sum(magnitude * mask, dim=(1, 2)) / torch.sum(mask[0])
    
    return band_magnitude

def evaluate_pipeline(pipeline_class, model_id, hf_token, model, config, num_images, batch_size):
    print(f"\n[+] Allocating VRAM for Generative Bottleneck: {model_id}...")
    try:
        pipeline = pipeline_class.from_pretrained(model_id, torch_dtype=torch.float16, token=hf_token, safety_checker=None).to(DEVICE)
    except Exception as e:
        print(f"HuggingFace Remote 503 Encountered. Attempting Native Local Cache Resolution...")
        pipeline = pipeline_class.from_pretrained(model_id, torch_dtype=torch.float16, token=hf_token, local_files_only=True, safety_checker=None).to(DEVICE)
    
    pipeline.set_progress_bar_config(disable=False)

    def purify(x):
        results_tensors = []
        for i in range(x.shape[0]):
            img = core_utils.tensor_to_pil(x[i])
            with torch.no_grad():
                purified_img = pipeline(prompt="", image=img, strength=0.35, guidance_scale=0.0).images[0]
            results_tensors.append(core_utils.pil_to_tensor(purified_img).to(DEVICE))
        return torch.stack(results_tensors)

    survival_ratios = {
        "patch": [],
        "natural": [],
        "grid": [],
        "grid_chromic": [],
        "grid_05": [],
        "grid_chromic_05": [],
        "grid_20": [],
        "grid_chromic_20": []
    }

    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    
    # Initialize all 4 proper attacks matching the original DiffPure script
    patch_atk = cc.AdvPatch(model=model, patch_size=config["purification"]["patch_size"])
    grid_atk = cc.TopologicalAttractor(eps=config["ThreatMappings"]["eps_structural"], channel="L")
    grid_chromic_atk = cc.TopologicalAttractor(eps=config["ThreatMappings"]["eps_structural"], channel="LC")
    natural_atk = cc.NarrowbandMimicry(eps=config["ThreatMappings"]["eps_structural"])
    
    # Initialize lower amplitude variants
    grid_atk_05 = cc.TopologicalAttractor(eps=0.05, channel="L")
    grid_chromic_atk_05 = cc.TopologicalAttractor(eps=0.05, channel="LC")
    grid_atk_20 = cc.TopologicalAttractor(eps=0.20, channel="L")
    grid_chromic_atk_20 = cc.TopologicalAttractor(eps=0.20, channel="LC")

    for offset in range(0, num_images, batch_size):
        curr_batch_size = min(batch_size, num_images - offset)
        print(f"\n  -> Processing Batch: Images {offset+1} to {offset+curr_batch_size} of {num_images} for {model_id}...")
        images, labels, _, _ = core_utils.load_imagenet_val_batch(curr_batch_size, offset=offset)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # 1. Apply Attacks
        adv_dict = {
            "patch": patch_atk(images, labels=labels),
            "natural": natural_atk(images, color_ops),
            "grid": grid_atk(images, color_ops),
            "grid_chromic": grid_chromic_atk(images, color_ops),
            "grid_05": grid_atk_05(images, color_ops),
            "grid_chromic_05": grid_chromic_atk_05(images, color_ops),
            "grid_20": grid_atk_20(images, color_ops),
            "grid_chromic_20": grid_chromic_atk_20(images, color_ops)
        }
        
        for atk_name, adv_images in adv_dict.items():
            # Route to the correct FFT extraction method based on topology
            if atk_name == "natural":
                pre_fft_mags = get_narrowband_magnitude(adv_images, color_ops)
                purified = purify(adv_images)
                post_fft_mags = get_narrowband_magnitude(purified, color_ops)
            else:
                pre_fft_mags = get_fft_peak_magnitude(adv_images, color_ops)
                purified = purify(adv_images)
                post_fft_mags = get_fft_peak_magnitude(purified, color_ops)
            
            
            ratio = (post_fft_mags / (pre_fft_mags + 1e-8)).cpu().tolist()
            survival_ratios[atk_name].extend(ratio)
        
        print(f"     => Batch Avg FFT Survival ({model_id} | Chromic Grid): {np.mean(survival_ratios['grid_chromic'][-curr_batch_size:])*100:.2f}%")

    # Wipe VRAM to prevent OOM when loading the next SDE
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return survival_ratios

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromacrypt_config.json"), "r") as f:
        config = json.load(f)
        
    num_images = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusion3Img2ImgPipeline
    core_utils.load_env()
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    
    print("Loading Targeting ResNet50 Classifier Baseline...")
    model = core_utils.load_victim_model().to(DEVICE)
    model.eval()

    print("\nBeginning Multi-Model Iterative FFT Amplitude Survival Extraction...")
    
    results = {}

    # Pass 1: DiffPure via SD 1.5
    results["sd15"] = evaluate_pipeline(
        StableDiffusionImg2ImgPipeline,
        "runwayml/stable-diffusion-v1-5",
        hf_token, model, config, num_images, batch_size
    )

    # Pass 2: DiffPure via SD 3.5
    results["sd35"] = evaluate_pipeline(
        StableDiffusion3Img2ImgPipeline,
        "stabilityai/stable-diffusion-3.5-medium",
        hf_token, model, config, num_images, batch_size
    )

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "diffpure_fft_survival_distribution_multi.json")
    with open(out_file, "w") as f:
        json.dump({
            "metadata": {
                "N": num_images,
                "metric": "Fourier (FFT) Amplitude Ratio at lambda=16 pixels (w=0.125pi)",
            },
            "survival_ratios": results,
            "mean_survival_percentages": {
                "sd15": {k: np.mean(v)*100 for k, v in results["sd15"].items()},
                "sd35": {k: np.mean(v)*100 for k, v in results["sd35"].items()}
            }
        }, f, indent=4)
        
    print(f"\nMulti-Model Continuous Distribution Export Completed Successfully to: {out_file}")

if __name__ == "__main__":
    main()
