import os
import sys
import gc
import warnings
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
warnings.simplefilter("ignore")
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
import torch
import torchvision.models as models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils
from diffusers import StableDiffusion3Img2ImgPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sd35_autoencoder_purify(img_tensor, pipeline):
    if pipeline is None: return img_tensor 
    print(f"    [Purification] Routing batch of {img_tensor.shape[0]} tensors simultaneously through SD 3.5 VAE...")
    
    pil_images = [core_utils.tensor_to_pil(img_tensor[i]) for i in range(img_tensor.shape[0])]
    
    with torch.no_grad():
        # Evaluate batch natively to prevent extreme PCIe bandwidth exhaustion per-image (CPU-to-VRAM swapping bottleneck)
        purified_images = pipeline(
            prompt=[""] * len(pil_images), 
            image=pil_images, 
            strength=0.35, 
            guidance_scale=0.0
        ).images
        
    results_tensors = [core_utils.pil_to_tensor(p).to(DEVICE) for p in purified_images]
    print(f"      -> Batch purification complete.")
    return torch.stack(results_tensors)

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
    
    pipeline = None
    try:
        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", 
            torch_dtype=torch.float16,
            token=hf_token
        )
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=False)
        print("SD 3.5 Large Flow-Matching Pipeline Loaded Successfully.")
    except Exception as e:
        print(f"Pipeline Loading Failed (Skipping Diffusion Purification): {e}")

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
        if pipeline:
            purified_wb = sd35_autoencoder_purify(adv_wb, pipeline)
            purified_bb = sd35_autoencoder_purify(adv_bb, pipeline)
            
            with torch.no_grad():
                preds_purified_wb = target((purified_wb - mean) / std).argmax(1)
                wb_sd35_wins += (preds_purified_wb != lbl_tensor).float().sum().item()
                
                preds_purified_bb = target((purified_bb - mean) / std).argmax(1)
                bb_sd35_wins += (preds_purified_bb != lbl_tensor).float().sum().item()
                
                
        total_imgs += curr_batch_size
        print(f"  -> Processed {total_imgs}/{num_images} validation tensors")
        
        # Prevent CUDA fragmentation graph buildup scaling into OOM death loops
        del adv_wb, adv_bb, chromic_atk_wb, chromic_atk_bb
        if pipeline:
            del purified_wb, purified_bb
        gc.collect()
        torch.cuda.empty_cache()

    asr_wb = (wb_wins/total_imgs)*100
    asr_bb = (bb_wins/total_imgs)*100
    asr_wb_sd35 = (wb_sd35_wins/total_imgs)*100 if pipeline else 0.0
    asr_bb_sd35 = (bb_sd35_wins/total_imgs)*100 if pipeline else 0.0

    print(f"\n--- [Final Validation Matrix] ---")
    print(f"WideResNet50 White-Box ASR: {asr_wb:.1f}%")
    print(f"WideResNet50 Black-Box (Transfer from ResNet50) ASR: {asr_bb:.1f}%")
    
    if pipeline:
        print(f"White-Box Post-Purification ASR: {asr_wb_sd35:.1f}%")
        print(f"Black-Box Post-Purification ASR: {asr_bb_sd35:.1f}%")

    print("\n" + "="*50)
    print("LaTeX Table [Robust Models] Synthesis Ready (Copy / Paste Below):")
    print("-" * 50)
    print(f"Threat Model & Robust Architecture & Pre-Purification ASR & Post-Purification (SD 3.5) ASR \\\\")
    print(f"\\midrule")
    if pipeline:
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
