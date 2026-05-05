import os
import sys
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import time
import numpy as np

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils
import lpips

# PyTorch 2.4+ Checkpoint Override
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

try:
    from robustbench.utils import load_model
except ImportError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromacrypt_config.json"), "r") as f:
    config = json.load(f)

# ------------- Structural Functions -------------

def pga_channel_ablation(images, labels, model, color_ops, freeze_L, freeze_C, freeze_H, steps=10, is_rgb=False):
    """
    [Logic Block]
    Operation: Targeted Channel Gamut Ablation (PGD Variation)
    Algebra:
      1. S_oklch = RGB_to_OKLCH(images)
      2. Computes L_{adv} = S_oklch + alpha_{l,c,h} * sign(grad(J))
      3. Imposes rigid scalar freeze checks: 
           If freeze_L: delta_L = 0 
           Else: delta_L = clip(delta_L, -eps_l, eps_l)
      4. Maps gradients seamlessly into independent visual boundaries
    Purpose: Mathematically paralyzes discrete adversarial vector dimensions to empirically measure the percentage of model collapse attributed exclusively to individual structural (L) versus color (C/H) layers.
    """
    normalize_fn = lambda x: (x - torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
    
    eps_rgb = config["ThreatMappings"]["eps_rgb"]
    eps_oklch_l = config["ThreatMappings"]["eps_oklch_l"]
    eps_oklch_c = config["ThreatMappings"]["eps_oklch_c"]
    eps_oklch_h = eps_oklch_l * 360.0  # Canonical equivalence: ChromicPGD convention
    
    if is_rgb:
        adv_rgb = images.clone().detach()
        adv_rgb.requires_grad = True
        alpha = eps_rgb / 4.0
        for _ in range(steps):
            loss = nn.CrossEntropyLoss()(model(normalize_fn(adv_rgb)), labels)
            model.zero_grad()
            loss.backward()
            if adv_rgb.grad is None: break
            adv_rgb.data = adv_rgb.data + alpha * adv_rgb.grad.sign()
            delta = torch.clamp(adv_rgb.data - images, -eps_rgb, eps_rgb)
            adv_rgb.data = torch.clamp(images + delta, 0, 1)
            adv_rgb.grad.zero_()
        with torch.no_grad(): preds_adv = model(normalize_fn(adv_rgb.detach())).argmax(dim=1)
        return (preds_adv != labels), adv_rgb.detach()
        
    img_oklch = color_ops.rgb_to_oklch(images)
    adv_oklch = img_oklch.clone().detach()
    adv_oklch.requires_grad = True
    alpha_l = eps_oklch_l / 4.0
    alpha_c = eps_oklch_c / 4.0
    alpha_h = eps_oklch_h / 4.0
    for _ in range(steps):
        adv_clipped = color_ops.gamut_clip_preserve_hue(adv_oklch, steps=12)
        adv_rgb = color_ops.oklch_to_rgb(adv_clipped).clamp(0, 1)
        loss = nn.CrossEntropyLoss()(model(normalize_fn(adv_rgb)), labels)
        model.zero_grad()
        loss.backward()
        if adv_oklch.grad is None: break
        
        adv_oklch.data[:, 0:1] = adv_oklch.data[:, 0:1] + alpha_l * adv_oklch.grad[:, 0:1].sign()
        adv_oklch.data[:, 1:2] = adv_oklch.data[:, 1:2] + alpha_c * adv_oklch.grad[:, 1:2].sign()
        adv_oklch.data[:, 2:3] = adv_oklch.data[:, 2:3] + alpha_h * adv_oklch.grad[:, 2:3].sign()
        
        delta = adv_oklch.data - img_oklch
        if freeze_L: delta[:, 0:1] = 0.0
        else: delta[:, 0:1] = torch.clamp(delta[:, 0:1], -eps_oklch_l, eps_oklch_l)
        if freeze_C: delta[:, 1:2] = 0.0
        else: delta[:, 1:2] = torch.clamp(delta[:, 1:2], -eps_oklch_c, eps_oklch_c)
        if freeze_H: delta[:, 2:3] = 0.0
        else: delta[:, 2:3] = torch.clamp(delta[:, 2:3], -eps_oklch_h, eps_oklch_h)
        
        adv_oklch.data = img_oklch + delta
        adv_oklch.data[:, 0:1] = torch.clamp(adv_oklch.data[:, 0:1], 0, 1)
        adv_oklch.data[:, 1:2] = torch.clamp(adv_oklch.data[:, 1:2], 0, 0.4)
        adv_oklch.data[:, 2:3] = adv_oklch.data[:, 2:3] % 360.0
        adv_oklch.grad.zero_()
        
    final_clipped = color_ops.gamut_clip_preserve_hue(adv_oklch, steps=12)
    img_adv = color_ops.oklch_to_rgb(final_clipped).clamp(0, 1).detach()
    with torch.no_grad(): preds_adv = model(normalize_fn(img_adv)).argmax(dim=1)
    return (preds_adv != labels), img_adv

def main():
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    loss_fn_lpips = lpips.LPIPS(net="vgg").to(DEVICE)
    
    num_images = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    
    print(f"Aggregating {num_images} base tensors gracefully via streaming logic...")
    images_cpu_list, labels_cpu_list = [], []
    for offset in range(0, num_images, batch_size):
        b_s = min(batch_size, num_images - offset)
        imgs, lbls, _, _ = core_utils.load_imagenet_val_batch(n_examples=b_s, offset=offset)
        images_cpu_list.append(imgs.cpu())
        labels_cpu_list.append(lbls.cpu())
    images = torch.cat(images_cpu_list)
    labels = torch.cat(labels_cpu_list)
    
    # Execution Mappings
    table5_data = {
        "RGB-PGD Baseline": {"freezes": (False, False, False), "is_rgb": True, "lpips": 0.0, "asr": {}, "mean_asr": 0.0},
        "OKLCH ($L+C+H$)": {"freezes": (False, False, False), "is_rgb": False, "lpips": 0.0, "asr": {}, "mean_asr": 0.0},
        "OKLCH ($L+C$) [Freeze $H$]": {"freezes": (False, False, True), "is_rgb": False, "lpips": 0.0, "asr": {}, "mean_asr": 0.0},
        "OKLCH ($C+H$) [Freeze $L$]": {"freezes": (True, False, False), "is_rgb": False, "lpips": 0.0, "asr": {}, "mean_asr": 0.0},
        "OKLCH ($L$) [Freeze $C+H$]": {"freezes": (False, True, True), "is_rgb": False, "lpips": 0.0, "asr": {}, "mean_asr": 0.0},
        "OKLCH ($C$) [Freeze $L+H$]": {"freezes": (True, False, True), "is_rgb": False, "lpips": 0.0, "asr": {}, "mean_asr": 0.0},
        "OKLCH ($H$) [Freeze $L+C$]": {"freezes": (True, True, False), "is_rgb": False, "lpips": 0.0, "asr": {}, "mean_asr": 0.0}
    }
    table9_data = {}
    table12_data = {}
    
    # TABLE 4 / TABLE 9 LOOP
    t9_models = config["full_benchmark"]["target_models"]
    for m_name in t9_models:
        print(f"\nEvaluating Core Logic onto [{m_name}]")
        try:
            if m_name == "resnet50":
                from torchvision import models
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE).eval()
                norm_fn = lambda x: (x - torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
            else:
                model = load_model(model_name=m_name, dataset='imagenet', threat_model='Linf').to(DEVICE).eval()
                norm_fn = lambda x: x
        except Exception as e:
            print(f"Skipping parameter '{m_name}' (Not strictly found in robustbench directory: {e})")
            continue
            
        rnd_16_fails, rnd_32_fails = 0, 0
        clean_correct = 0
        nat_fails, gl_fails, glc_fails = 0, 0, 0
        patch_fails = 0
        rnd_fails, pgd_fails = 0, 0
        total_glc_lpips = 0.0
        total_nat_lpips, total_gl_lpips = 0.0, 0.0
        total_nat_ssim, total_gl_ssim, total_glc_ssim = 0.0, 0.0, 0.0
        
        for b_idx in range(0, len(images), batch_size):
            sys.stdout.write(f"\r  -> Executing {m_name} | Batch [{b_idx//batch_size + 1}/{(len(images)+batch_size-1)//batch_size}] ({b_idx}/{len(images)} images) ... ")
            sys.stdout.flush()
            b_img = images[b_idx:b_idx+batch_size].to(DEVICE)
            b_lbl = labels[b_idx:b_idx+batch_size].to(DEVICE)
            
            with torch.no_grad():
                c_preds = model(norm_fn(b_img)).argmax(dim=1)
                clean_mask = (c_preds == b_lbl)
                clean_correct += clean_mask.sum().item()
                
            eps_s = config["ThreatMappings"]["eps_structural"]
            
            atk_nat = cc.NarrowbandMimicry(eps=eps_s)
            atk_glc = cc.TopologicalAttractor(eps=eps_s, channel="LC")
            atk_patch = cc.AdvPatch(model=model, patch_size=config["purification"]["patch_size"])
            
            adv_nat = atk_nat(b_img, color_ops)
            adv_glc = atk_glc(b_img, color_ops)
            adv_patch = atk_patch(b_img, labels=b_lbl)
            
            # L-only grid based on topological base
            grid_base = cc.generate_topological_grid(b_img.size(2), b_img.size(3), DEVICE)
            img_oklch = color_ops.rgb_to_oklch(b_img)
            L, C, H_chan = img_oklch[:, 0:1, :, :], img_oklch[:, 1:2, :, :], img_oklch[:, 2:3, :, :]
            L_gl = (L + eps_s * grid_base).clamp(0, 1)
            adv_gl = color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(torch.cat([L_gl, C, H_chan], dim=1), steps=12)).clamp(0, 1).detach()
            
            # Random L noise
            noise_rnd = (torch.rand_like(L, device=DEVICE) * 2 - 1)
            L_rnd = (L + eps_s * noise_rnd).clamp(0, 1)
            adv_rnd = color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(torch.cat([L_rnd, C, H_chan], dim=1), steps=12)).clamp(0, 1).detach()
            
            with torch.no_grad():
                m_nat = (model(norm_fn(adv_nat)).argmax(dim=1) != b_lbl)
                m_gl = (model(norm_fn(adv_gl)).argmax(dim=1) != b_lbl)
                m_glc = (model(norm_fn(adv_glc)).argmax(dim=1) != b_lbl)
                m_rnd = (model(norm_fn(adv_rnd)).argmax(dim=1) != b_lbl)
                m_patch = (model(norm_fn(adv_patch)).argmax(dim=1) != b_lbl)
            
            m_pgd, adv_pgd = pga_channel_ablation(b_img, b_lbl, model, color_ops, False, False, False, steps=config["ablation"]["pgd_steps"])
            
            with torch.no_grad(): 
                total_nat_lpips += loss_fn_lpips(b_img * 2 - 1, adv_nat * 2 - 1).mean().item() * b_img.size(0)
                total_gl_lpips += loss_fn_lpips(b_img * 2 - 1, adv_gl * 2 - 1).mean().item() * b_img.size(0)
                total_glc_lpips += loss_fn_lpips(b_img * 2 - 1, adv_glc * 2 - 1).mean().item() * b_img.size(0)
            
            nat_fails += m_nat[clean_mask].sum().item()
            gl_fails += m_gl[clean_mask].sum().item()
            glc_fails += m_glc[clean_mask].sum().item()
            rnd_fails += m_rnd[clean_mask].sum().item()
            pgd_fails += m_pgd[clean_mask].sum().item()
            patch_fails += m_patch[clean_mask].sum().item()
            
            print(f"| Clean Acc: {clean_mask.sum().item()}/{b_img.size(0)} | Patch Fails: {m_patch[clean_mask].sum().item()} | Grid Fails: {m_glc[clean_mask].sum().item()}")
            
        c_acc = clean_correct / num_images
        
        table9_data[m_name] = {
            "Clean_Accuracy": c_acc,
            "Natural_ASR": nat_fails / clean_correct if clean_correct > 0 else 0,
            "Natural_LPIPS": total_nat_lpips / num_images,
            "Natural_SSIM": 0.0,
            "Grid_L_ASR": gl_fails / clean_correct if clean_correct > 0 else 0,
            "Grid_L_LPIPS": total_gl_lpips / num_images,
            "Grid_L_SSIM": 0.0,
            "Grid_LC_ASR": glc_fails / clean_correct if clean_correct > 0 else 0,
            "Grid_LC_LPIPS": total_glc_lpips / num_images,
            "Grid_LC_SSIM": 0.0,
            "Total": num_images
        }
        
        table12_data[m_name] = {
            "Clean_Accuracy": c_acc,
            "Adv_Patch_ASR": patch_fails / clean_correct if clean_correct > 0 else 0,
            "Narrowband_ASR": nat_fails / clean_correct if clean_correct > 0 else 0,
            "Grid_Harmonic_ASR": glc_fails / clean_correct if clean_correct > 0 else 0,
            "Avg_LPIPS_Footprint": total_glc_lpips / num_images
        }
        
    with open(os.path.join(RESULTS_DIR, "table3_full_benchmark.json"), "w") as f: json.dump({"metadata":{"N": num_images}, "results": table9_data}, f, indent=4)    
    with open(os.path.join(RESULTS_DIR, "table10_structured_baselines.json"), "w") as f: json.dump(table12_data, f, indent=4)

    # TABLE 5 LOOP
    t5_models = config["ablation"]["target_models"]
    for map_key, map_conf in table5_data.items():
        fL, fC, fH = map_conf["freezes"]
        total_lpips = 0.0
        is_rgb = map_conf.get("is_rgb", False)
        for m_name in t5_models:
            print(f" -> Mapping to {m_name}")
            try:
                if m_name == "resnet50":
                    from torchvision.models import resnet50, ResNet50_Weights
                    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE).eval()
                elif m_name == "vit_b_16":
                    from torchvision.models import vit_b_16, ViT_B_16_Weights
                    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(DEVICE).eval()
                elif m_name == "vgg16":
                    from torchvision.models import vgg16, VGG16_Weights
                    model = vgg16(weights=VGG16_Weights.DEFAULT).to(DEVICE).eval()
                elif m_name == "densenet121":
                    from torchvision.models import densenet121, DenseNet121_Weights
                    model = densenet121(weights=DenseNet121_Weights.DEFAULT).to(DEVICE).eval()
                elif m_name == "efficientnet_b0":
                    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(DEVICE).eval()
                else: continue
            except Exception as e: continue
                
            fails = 0
            for b_idx in range(0, len(images), batch_size):
                b_img = images[b_idx:b_idx+batch_size].to(DEVICE)
                b_lbl = labels[b_idx:b_idx+batch_size].to(DEVICE)
                succ, adv_out = pga_channel_ablation(b_img, b_lbl, model, color_ops, fL, fC, fH, steps=config["ablation"]["pgd_steps"], is_rgb=is_rgb)
                fails += succ.sum().item()
                if m_name == t5_models[0]:  # Only compile LPIPS once per geometric trace
                    with torch.no_grad(): total_lpips += loss_fn_lpips(b_img * 2 - 1, adv_out * 2 - 1).mean().item() * b_img.size(0)
                        
            name_map = {
                "resnet50": "ResNet50",
                "vit_b_16": "ViT-B-16",
                "vgg16": "VGG16",
                "densenet121": "DenseNet121",
                "efficientnet_b0": "EfficientNet"
            }
            table5_data[map_key]["asr"][name_map.get(m_name, m_name)] = (fails / num_images) * 100
        table5_data[map_key]["lpips"] = total_lpips / num_images
        model_asrs = list(table5_data[map_key]["asr"].values())
        table5_data[map_key]["mean_asr"] = sum(model_asrs) / len(model_asrs) if model_asrs else 0.0
        
    # Re-structure for JSON omitting functions
    out_table5 = {k: {"lpips": v["lpips"], "asr": v["asr"], "mean_asr": v["mean_asr"]} for k,v in table5_data.items()}
    with open(os.path.join(RESULTS_DIR, "table1_channel_ablation.json"), "w") as f: json.dump(out_table5, f, indent=4)
    
    print("\nOrchestrator Sequence Complete. JSON traces successfully dumped. Execute 'generate_latex_tables.py' hook to finalize matrices.")

if __name__ == "__main__":
    main()
