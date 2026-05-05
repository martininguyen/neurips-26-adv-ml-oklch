import torch
import torch.nn as nn
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json

import chromacrypt_module as cc
from chromacrypt_module import utils as core_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    """
    [Logic Block]
    Operation: Computational Resource & Scalability Benchmarking
    Algebra:
      1. rgb_time = Runtime(RGB_PGD) / Iterations
      2. oklch_time = Runtime(OKLCH_PGD) / Iterations
      3. Slowdown Factor = oklch_time / rgb_time
      4. Extrapolates max_memory_allocated bounds natively
    Purpose: Empirically measures architectural processing latency natively introduced by cylindrical-cartesian backpropagation dependencies vs native linear RGB constraints.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = core_utils.load_victim_model().to(DEVICE).eval()
    color_ops = cc.DifferentiableColorOps().to(DEVICE)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    
    # Generate dummy batch
    with open(os.path.join(script_dir, "..", "chromacrypt_config.json")) as f: config = json.load(f)
    batch_size = config["dataset"]["batch_size"]
    images = torch.rand(batch_size, 3, 224, 224, device=DEVICE)
    labels = torch.randint(0, 1000, (batch_size,), device=DEVICE)
    
    eps_rgb = config["ThreatMappings"]["eps_rgb"]
    eps_l = config["ThreatMappings"]["eps_oklch_l"]
    eps_c = config["ThreatMappings"]["eps_oklch_c"]
    eps_h = eps_l * 360.0
    
    # Warmup
    print("Warming up CUDA...")
    for _ in range(5):
        RGB_atk = cc.RGBPGD(model=model, eps=eps_rgb)
        _ = RGB_atk(images, labels)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark RGB
    start = time.time()
    steps = config["ablation"].get("pgd_steps", 10)
    for _ in range(steps):
        _ = RGB_atk(images, labels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        rgb_time = ((time.time() - start) / (steps * 10)) * 1000 # per iteration inside PGD
        rgb_mem = torch.cuda.max_memory_allocated() / (1024*1024)
    else:
        rgb_time = ((time.time() - start) / (steps * 10)) * 1000
        rgb_mem = 0
    
    # Warmup OKLCH
    for _ in range(5):
        OKLCH_atk = cc.ChromicPGD(model=model, eps_l=eps_l, eps_c=eps_c, eps_h=eps_h)
        _ = OKLCH_atk(images, labels)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark OKLCH
    start = time.time()
    for _ in range(steps):
        _ = OKLCH_atk(images, labels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        oklch_time = ((time.time() - start) / (steps * 10)) * 1000 # per iteration inside PGD
        oklch_mem = torch.cuda.max_memory_allocated() / (1024*1024)
    else:
        oklch_time = ((time.time() - start) / (steps * 10)) * 1000
        oklch_mem = 0
    
    results = {
        "RGB-PGD (ms/iter)": round(rgb_time, 2),
        "OKLCH-PGD (ms/iter)": round(oklch_time, 2),
        "Slowdown Factor": round(oklch_time / rgb_time, 2) if rgb_time > 0 else 0,
        "Device": str(DEVICE),
        "Peak GPU Memory (MB)": round(max(rgb_mem, oklch_mem), 1)
    }
    
    print("\n[ LaTeX Table [Overhead Diagnostics] Synthesis ]")
    print(json.dumps(results, indent=4))
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table7_overhead.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("LaTeX Table [Overhead Diagnostics] export completed.")

if __name__ == "__main__":
    main()
