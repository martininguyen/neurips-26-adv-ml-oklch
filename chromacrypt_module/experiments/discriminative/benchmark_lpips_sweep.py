import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import os
import json
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from robustbench.utils import load_model
import lpips

from chromacrypt_module.color_ops import DifferentiableColorOps
from chromacrypt_module import utils as core_utils
import chromacrypt_module as cc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromacrypt_config.json"), "r") as f:
    SETTINGS = json.load(f)

def generate_adv_from_noise(img_tensor, color_ops, channel='L', noise=None, amp=0.5):
    img_oklch = color_ops.rgb_to_oklch(img_tensor)
    L = img_oklch[:, 0:1, :, :]
    L_adv = (L + amp * noise).clamp(0, 1)
    adv_oklch = torch.cat([L_adv, img_oklch[:, 1:2, :, :], img_oklch[:, 2:3, :, :]], dim=1)
    return color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(adv_oklch, steps=12)).clamp(0, 1)

def match_lpips(img_tensor, loss_fn_vgg, color_ops, target_lpips, mode='rgb', base_noise=None):
    """
    [Logic Block]
    Operation: Granular LPIPS Threshold Sweeping
    Algebra:
      1. mid = (low + high) / 2
      2. If target > current LPIPS, scale epsilon boundary upward (low = mid).
      3. Validates boundary alignment against discrete visual threshold sets [0.2, 0.4, 0.6, 0.8].
    Purpose: Ensures standard deviation metrics map smoothly across continuously increasing topological disruption matrices without failing linearly.
    """
    low, high = 0.0, 5.0
    best_adv = None
    for _ in range(25):
        mid = (low + high) / 2.0
        if mode == 'rgb':
            candidate = (img_tensor + mid * base_noise).clamp(0, 1)
        else:
            candidate = generate_adv_from_noise(img_tensor, color_ops, 'L', base_noise, mid)
        
        current_lpips = loss_fn_vgg(img_tensor * 2 - 1, candidate * 2 - 1).item()
        best_adv = candidate
        if abs(current_lpips - target_lpips) < 0.015:
            return best_adv
        elif current_lpips < target_lpips:
            low = mid
        else:
            high = mid
    return best_adv

def main():
    color_ops = DifferentiableColorOps().to(DEVICE)
    loss_fn_vgg = lpips.LPIPS(net="alex").to(DEVICE)
    try:
        model = load_model(model_name="Liu2023Comprehensive_ConvNeXt-L", dataset='imagenet', threat_model='Linf').to(DEVICE).eval()
        normalize = lambda x: x
    except Exception as e:
        print(f"Critial Loading Failure for Liu2023: {e}\nAborting execution.")
        return

    data_dir = core_utils.find_imagenet_dir()
    import glob
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.JPEG")) + glob.glob(os.path.join(data_dir, "*.jpg")))
    test_images = all_files[:SETTINGS["dataset"]["num_test_images"]]
    
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    thresholds = [0.2, 0.4, 0.6, 0.8]
    stats = {str(th): {"grid_fail": 0, "rgb_fail": 0, "count": 0} for th in thresholds}
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    for i, img_p in enumerate(test_images):
        try: img = Image.open(img_p).convert('RGB')
        except: continue
        img_tensor = t(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_clean = model(normalize(img_tensor)).argmax(dim=1).item()
            
            _, _, h, w = img_tensor.shape
            grid_noise = cc.generate_topological_grid(h, w, DEVICE)
            rgb_noise = (torch.rand_like(img_tensor) * 2 - 1).to(DEVICE)
            
            for th in thresholds:
                adv_grid = match_lpips(img_tensor, loss_fn_vgg, color_ops, th, 'grid', grid_noise)
                adv_rgb = match_lpips(img_tensor, loss_fn_vgg, color_ops, th, 'rgb', rgb_noise)
                
                if model(normalize(adv_grid)).argmax(dim=1).item() != pred_clean:
                    stats[str(th)]["grid_fail"] += 1
                if model(normalize(adv_rgb)).argmax(dim=1).item() != pred_clean:
                    stats[str(th)]["rgb_fail"] += 1
                stats[str(th)]["count"] += 1
                
        if i % 10 == 0:
            print(f"Processed {i+1}: {stats}")

    print("Final Output:")
    print(json.dumps(stats, indent=4))
    
    with open(os.path.join(out_dir, "lpips_sweep.json"), "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
