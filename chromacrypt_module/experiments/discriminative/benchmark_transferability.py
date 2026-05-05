import torch
import torchvision.models as models
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
import lpips

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "chromacrypt_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    num_images = config["dataset"]["num_test_images"]
    batch_size = config["dataset"]["batch_size"]
    eps_rgb = config.get("ThreatMappings", {}).get("eps_rgb", 0.03137)
    # True OKLCH equivalents map to 0.01 absolutely natively, not 8/255 conversion
    eps_l = config.get("ThreatMappings", {}).get("eps_oklch_l", 0.01)
    eps_c = config.get("ThreatMappings", {}).get("eps_oklch_c", 0.01)
    eps_h = eps_l * 360.0 # Bounded to 3.6 degrees equivalently

    print(f"Initializing ChromaCrypt Transferability Framework (N={num_images})...")
    surrogate = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().to(DEVICE)
    
    targets = {}
    for t in config["transferability"]["target_models"]:
        if t == "alexnet": targets["AlexNet"] = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).eval().to(DEVICE)
        elif t == "vgg16": targets["VGG16"] = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval().to(DEVICE)
        elif t == "mobilenet_v3_large": targets["MobileNet"] = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).eval().to(DEVICE)
        elif t == "efficientnet_b0": targets["EfficientNet"] = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).eval().to(DEVICE)
        elif t == "vit_b_16": targets["ViT-B-16"] = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).eval().to(DEVICE)
    
    loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)
    
    attacks = {
        "RGB-PGD Baseline": cc.RGBPGD(model=surrogate, eps=eps_rgb), 
        "OKLCH ($L+C+H$)": cc.ChromicPGD(model=surrogate, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h, freeze_L=False, freeze_C=False, freeze_H=False),
        "OKLCH ($L+C$) [Freeze $H$]": cc.ChromicPGD(model=surrogate, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h, freeze_L=False, freeze_C=False, freeze_H=True),
        "OKLCH ($C+H$) [Freeze $L$]": cc.ChromicPGD(model=surrogate, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h, freeze_L=True, freeze_C=False, freeze_H=False),
        "OKLCH ($L$) [Freeze $C+H$]": cc.ChromicPGD(model=surrogate, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h, freeze_L=False, freeze_C=True, freeze_H=True),
        "OKLCH ($C$) [Freeze $L+H$]": cc.ChromicPGD(model=surrogate, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h, freeze_L=True, freeze_C=False, freeze_H=True),
        "OKLCH ($H$) [Freeze $L+C$]": cc.ChromicPGD(model=surrogate, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h, freeze_L=True, freeze_C=True, freeze_H=False),
    }
    
    results = {atk: {"lpips": 0.0, "asr": {t: 0 for t in targets}, "mean_asr": 0.0} for atk in attacks}
    target_clean_counts = {t: 0 for t in targets}
    total_lpips_count = 0

    print("Executing iterative memory-safe evaluation loop (Surrogate: ResNet50)...")
    for offset in range(0, num_images, batch_size):
        curr_batch_size = min(batch_size, num_images - offset)
        images, labels, mean, std = core_utils.load_imagenet_val_batch(n_examples=curr_batch_size, offset=offset)
        img_tensor = images.to(DEVICE)
        lbl_tensor = labels.to(DEVICE)
        
        # Verify clean mask intersecting with target domains strictly eliminating inherent misclassifications
        with torch.no_grad():
            clean_preds_surr = surrogate((img_tensor - mean)/std).argmax(dim=1)
            clean_mask_surr = (clean_preds_surr == lbl_tensor)
            total_lpips_count += clean_mask_surr.sum().item()
            
            target_joint_masks = {}
            for t_name, t_model in targets.items():
                t_clean_preds = t_model((img_tensor - mean)/std).argmax(dim=1)
                joint_mask = clean_mask_surr & (t_clean_preds == lbl_tensor)
                target_joint_masks[t_name] = joint_mask
                target_clean_counts[t_name] += joint_mask.sum().item()
        
        if clean_mask_surr.sum().item() == 0: continue

        for atk_name, atk_fn in attacks.items():
            adv_img = atk_fn(img_tensor, lbl_tensor)
            """
            [Logic Block]
            Operation: Black-box Multi-Architecture Surrogate Transfer Validation
            Algebra:
               1. G_adv = Optimization(Surrogate, Image) -> ResNet50 Matrix Boundaries
               2. For Target in {VGG, EfficientNet, ViT, MobileNet}:
                     Preds_t = Target(G_adv)
               3. If Preds_t != Labels: Count Transferability Logic Flaw.
            Purpose: Mathematically assesses whether the adversarial boundary manipulation targets localized topological anomalies specific only to the ResNet mapping parameter space, or isolates continuous geometric vulnerabilities spanning universal hierarchical architectures.
            """
            with torch.no_grad():
                results[atk_name]["lpips"] += loss_fn(img_tensor[clean_mask_surr] * 2.0 - 1.0, adv_img[clean_mask_surr] * 2.0 - 1.0).sum().item()
                adv_input = (adv_img - mean) / std
                
                for t_name, t_model in targets.items():
                    j_mask = target_joint_masks[t_name]
                    if j_mask.sum().item() > 0:
                        preds = t_model(adv_input[j_mask]).argmax(dim=1)
                        results[atk_name]["asr"][t_name] += (preds != lbl_tensor[j_mask]).float().sum().item()
            print(f"  -> Evaluated {offset + curr_batch_size}/{num_images} base datasets")

    # Aggregate
    for atk_name in results:
        results[atk_name]["lpips"] /= total_lpips_count if total_lpips_count > 0 else 1
        for t_name in targets:
            results[atk_name]["asr"][t_name] = (results[atk_name]["asr"][t_name] / target_clean_counts[t_name]) * 100 if target_clean_counts[t_name] > 0 else 0
        results[atk_name]["mean_asr"] = sum(results[atk_name]["asr"].values()) / len(targets)

    print("\n" + "="*80)
    print("LaTeX Table [Transferability] Synthesis Ready:")
    print("-" * 80)
    print(f"{'Optimization Domain':<30} | {'AlexNet':>8} | {'VGG16':>8} | {'MobileNet':>9} | {'EfficientNet':>12} | {'ViT-B-16':>8} | {'Mean ASR':>8}")
    print("-" * 80)
    for atk_name, metrics in results.items():
        alex = f"{metrics['asr'].get('AlexNet', 0):.1f}%"
        vgg = f"{metrics['asr'].get('VGG16', 0):.1f}%"
        mob = f"{metrics['asr'].get('MobileNet', 0):.1f}%"
        eff = f"{metrics['asr'].get('EfficientNet', 0):.1f}%"
        vit = f"{metrics['asr'].get('ViT-B-16', 0):.1f}%"
        mean = f"{metrics['mean_asr']:.1f}%"
        print(f"{atk_name:<30} | {alex:>8} | {vgg:>8} | {mob:>9} | {eff:>12} | {vit:>8} | {mean:>8}")
    print("="*80 + "\n")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    with open(os.path.join(results_dir, 'table8_transferability.json'), 'w') as f:
        json.dump({"metadata": {"surrogate_model": "ResNet50", "N": num_images}, "results": results}, f, indent=4)
        
    print("LaTeX Table [Transferability] Generation Complete. Data formatted natively.")
    
if __name__ == "__main__":
    main()
