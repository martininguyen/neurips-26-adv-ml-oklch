import os
import sys
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import time
import torchattacks

try:
    from autoattack import AutoAttack
except ImportError:
    print("CRITICAL: AutoAttack must be installed via `pip install git+https://github.com/fra31/auto-attack`")
    sys.exit(1)

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils
import lpips

# RobustBench Integration
from robustbench.utils import load_model

# PyTorch 2.4+ Checkpoint Override
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

# RobustBench ImageNet "model." prefix bug workaround for Timm Architectures (Swin-L)
import robustbench.utils as rb_utils
_original_safe_load_state_dict = rb_utils._safe_load_state_dict
def _smart_load_state_dict(model, model_name, state_dict, dataset_):
    try:
        return _original_safe_load_state_dict(model, model_name, state_dict, dataset_)
    except RuntimeError as getattr_e:
        if "Missing key(s)" in str(getattr_e) and "Unexpected key(s)" in str(getattr_e):
            clean_dict = { (k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items() }
            # Bypass strict temporarily to let Timm architectures ingest flat state mappings
            model.load_state_dict(clean_dict, strict=False)
            return model
        raise getattr_e
rb_utils._safe_load_state_dict = _smart_load_state_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NormModelWrapper(nn.Module):
    """Wraps model to include normalization inside the forward pass for standardized PGD"""
    def __init__(self, model, robustbench_native=True):
        super().__init__()
        self.model = model
        self.robustbench_native = robustbench_native
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)

    def forward(self, x):
        if self.robustbench_native:
            return self.model(x)
        else:
            return self.model((x - self.mean) / self.std)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "chromacrypt_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    num_images = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    target_models = config["autoattack"]["target_models"]
    eps_rgb = config["ThreatMappings"]["eps_autoattack_rgb"]
    eps_oklch = config["ThreatMappings"]["eps_oklch_c"]
    
    print(f"Loading ChromaCrypt PGD Validation Validation (N={num_images})...", flush=True)
    
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    loss_fn = lpips.LPIPS(net="alex").to(DEVICE)
    
    # Pre-caching Images to CPU RAM to prevent CUDA Out-of-Memory Lockups
    images_cpu_list = []
    labels_cpu_list = []
    
    for offset in range(0, num_images, batch_size):
        curr_batch_size = min(batch_size, num_images - offset)
        imgs, lbls, _, _ = core_utils.load_imagenet_val_batch(n_examples=curr_batch_size, offset=offset)
        images_cpu_list.append(imgs.cpu())
        labels_cpu_list.append(lbls.cpu())
        
    images = torch.cat(images_cpu_list)
    labels = torch.cat(labels_cpu_list)
    
    print(f"Aggregated {len(images)} System RAM images seamlessly.", flush=True)
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    json_path = os.path.join(results_dir, "table10_autoattack_comparison.json")
    csv_path = os.path.join(results_dir, "table10_autoattack_comparison.csv")
    
    results = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                saved_data = json.load(f)
                results = saved_data.get("results", [])
            print(f"Loaded {len(results)} existing model results from checkpoint.", flush=True)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}", flush=True)
            
    completed_models = set(r["model"] for r in results)
    
    for model_id in target_models:
        if model_id in completed_models:
            print(f"\n--- Skipping {model_id} (already evaluated in checkpoint) ---", flush=True)
            continue
            
        print(f"\n--- Loading {model_id} ---", flush=True)
        robustbench_native = True
        try:
            if model_id == "resnet50":
                from torchvision import models
                model = models.resnet50(pretrained=True).to(DEVICE).eval()
                robustbench_native = False
            else:
                model = load_model(model_name=model_id, dataset='imagenet', threat_model='Linf').to(DEVICE).eval()
        except Exception as e:
            print(f"Model '{model_id}' failed to load: {e}\nAborting execution on {model_id}.", flush=True)
            continue

        print("Running SOTA AutoAttack (RGB Bounds)...", flush=True)
        aa_wrapper_rgb = NormModelWrapper(model, robustbench_native=robustbench_native)
        
        print("Scoring Clean Tensors...", flush=True)
        with torch.no_grad():
            preds_clean_list = []
            for b_idx in range(0, len(images), batch_size):
                b_img = images[b_idx:b_idx+batch_size].to(DEVICE)
                b_pred = aa_wrapper_rgb(b_img).argmax(dim=1)
                preds_clean_list.append(b_pred.cpu())
            preds_clean = torch.cat(preds_clean_list)
            
        adversary = AutoAttack(aa_wrapper_rgb, norm='Linf', eps=eps_rgb, version='standard', device=DEVICE)
        
        # Aggressive memory constraint to prevent graphic VRAM lockups during APGD matrix operations
        optimal_aa_bs = min(batch_size, 4)
        
        start = time.time()
        adv_aa = adversary.run_standard_evaluation(images.to(DEVICE), preds_clean.to(DEVICE), bs=optimal_aa_bs).cpu()
        print(f"RGB Attack Computed in {time.time()-start:.1f}s", flush=True)
        
        rgb_wins = 0
        lpips_rgb = 0.0
        
        with torch.no_grad():
            for b_idx in range(0, len(images), batch_size):
                b_img = images[b_idx:b_idx+batch_size].to(DEVICE)
                b_adv = adv_aa[b_idx:b_idx+batch_size].to(DEVICE)
                b_lbl = preds_clean[b_idx:b_idx+batch_size].to(DEVICE)
                
                b_pred = aa_wrapper_rgb(b_adv).argmax(dim=1)
                rgb_wins += (b_pred != b_lbl).float().sum().item()
                lpips_rgb += loss_fn(b_img, b_adv).sum().item()
                
        asr_rgb = (rgb_wins / num_images) * 100
        avg_lpips_rgb = lpips_rgb / num_images
        
        print("Running SOTA AutoAttack (OKLCH Native Bounds)...", flush=True)
        
        class CustomOKLCHWrapper(core_utils.OKLCHModelWrapper):
            def forward(self, x_norm):
                oklch_input = self.unscale(x_norm)
                clipped_oklch = self.color_ops.gamut_clip_preserve_hue(oklch_input, steps=12)
                rgb_out = self.color_ops.oklch_to_rgb(clipped_oklch).contiguous().clamp(0.0, 1.0)
                if robustbench_native:
                    return self.base_model(rgb_out)
                else:
                    return self.base_model((rgb_out - self.mean) / self.std)

        oklch_model = CustomOKLCHWrapper(model, freeze_H=True).to(DEVICE)
        
        with torch.no_grad():
            batch_oklch = color_ops.rgb_to_oklch(images.to(DEVICE))
            L = batch_oklch[:, 0:1]
            C = batch_oklch[:, 1:2] / 0.4
            H = batch_oklch[:, 2:3] / 360.0
            images_oklch_norm = torch.cat([L, C, H], dim=1).clamp(0, 1).cpu()
            
        oklch_model.clean_oklch_norm = images_oklch_norm.to(DEVICE)
        adversary_oklch = AutoAttack(oklch_model, norm='Linf', eps=eps_rgb, version='standard', device=DEVICE)
        
        start_ok = time.time()
        adv_aa_oklch_norm = adversary_oklch.run_standard_evaluation(images_oklch_norm.to(DEVICE), preds_clean.to(DEVICE), bs=optimal_aa_bs).cpu()
        print(f"OKLCH Attack Computed in {time.time()-start_ok:.1f}s", flush=True)
        
        oklch_wins = 0
        lpips_oklch = 0.0
        
        with torch.no_grad():
            for b_idx in range(0, len(images), batch_size):
                b_img = images[b_idx:b_idx+batch_size].to(DEVICE)
                b_adv_norm = adv_aa_oklch_norm[b_idx:b_idx+batch_size].to(DEVICE)
                b_lbl = preds_clean[b_idx:b_idx+batch_size].to(DEVICE)
                
                b_adv_raw = oklch_model.unscale(b_adv_norm)
                b_adv_rgb = color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(b_adv_raw, steps=12)).clamp(0, 1)
                
                b_pred = oklch_model(b_adv_norm).argmax(dim=1)
                oklch_wins += (b_pred != b_lbl).float().sum().item()
                lpips_oklch += loss_fn(b_img, b_adv_rgb).sum().item()
                
        asr_oklch = (oklch_wins / num_images) * 100
        avg_lpips_oklch = lpips_oklch / num_images
        
        results.append({
            "model": model_id,
            "RGB_ASR": asr_rgb,
            "RGB_LPIPS": avg_lpips_rgb,
            "OKLCH_ASR": asr_oklch,
            "OKLCH_LPIPS": avg_lpips_oklch
        })

        # JSON Dump Hook (Incremental Save)
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        with open(json_path, "w") as f:
            json.dump({"results": results}, f, indent=4)
            
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Attack", "ASR (%)", "Avg LPIPS"])
            for res in results:
                writer.writerow([res["model"], "AutoAttack (RGB)", f"{res['RGB_ASR']:.1f}", f"{res['RGB_LPIPS']:.4f}"])
                writer.writerow([res["model"], "AutoAttack (OKLCH)", f"{res['OKLCH_ASR']:.1f}", f"{res['OKLCH_LPIPS']:.4f}"])

if __name__ == "__main__":
    main()
