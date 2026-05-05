"""
Summary: Generates assets for qualitative AB-testing and human perception studies.
Objective: Produces identical image pairs (Clean vs. Perturbed) for manual verification, ensuring biological operators cannot perceptually distinguish the OKLCH structural shift.
Execution: Executes fine-tuned low-epsilon attacks and aggregates outputs into structured asset directories.
"""


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys
import glob
import random
import csv
import json

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "experiment_human_study_gen.json"), "r") as f:
    SETTINGS = json.load(f)

# Ensure we can import the color ops
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "working_scripts"))
    sys.path.append(os.path.join(project_root, "oklch_defense"))
try:
    from chromacrypt_module.color_ops import DifferentiableColorOps
    from chromacrypt_module.attacks import ChromicPGD
except ImportError:
    print("Error: chromacrypt_module not found.")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Removed inline apply_oklch_pgd in favor of canonical ChromicPGD

def create_pair(clean_pil, attacked_pil, pair_id, output_dir):
    """
    Creates a side-by-side comparison image (randomized L/R).
    Returns 'Left' or 'Right' indicating where the CLEAN image is.
    """
    # Resize for standardization
    target_size = (512, 512)
    clean_rez = clean_pil.resize(target_size)
    attacked_rez = attacked_pil.resize(target_size)
    
    # Randomize
    is_clean_left = random.choice([True, False])
    
    # Create canvas
    gap = 20
    canvas_w = target_size[0] * 2 + gap
    canvas_h = target_size[1]
    
    pair_img = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    
    if is_clean_left:
        pair_img.paste(clean_rez, (0, 0))
        pair_img.paste(attacked_rez, (target_size[0] + gap, 0))
        correct_answer = "Left"
    else:
        pair_img.paste(attacked_rez, (0, 0))
        pair_img.paste(clean_rez, (target_size[0] + gap, 0))
        correct_answer = "Right"
        
    # Save
    filename = f"pair_{pair_id:03d}.png"
    pair_img.save(os.path.join(output_dir, filename))
    
    return filename, correct_answer

def main():
    num_samples = SETTINGS["dataset"]["num_samples"]
    output_dir = SETTINGS["dataset"]["output_dir"]
    
    # Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {num_samples} study pairs in '{output_dir}'...")
    
    # Load Model (Needed for Gradient Calculation)
    print("Loading ResNet50 for Gradient Generation...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE).eval()
    
    color_ops = DifferentiableColorOps().to(DEVICE)
    
    # Load ImageNet images
    img_dir = os.path.join(project_root, "data", "imagenet-1k")
        
    all_files = sorted(glob.glob(os.path.join(img_dir, "*.JPEG")))
    if not all_files:
        print("Error: No ImageNet images found.")
        return

    # Randomly select diverse images
    selected_indices = random.sample(range(len(all_files)), min(num_samples, len(all_files)))
    
    t = transforms.Compose([
        transforms.Resize((512, 512)), # Use higher res for study
        transforms.ToTensor()
    ])
    
    # ImageNet Mean/Std for Model Normalization
    # Note: PGD usually runs on [0,1], but model needs Norm.
    # We will wrap normalization into the forward pass logic or handle it carefully.
    # Actually models.resnet50 expects normalized input.
    # Let's create a Normalize wrapper to keep the PGD logic clean (PGD sees 0-1, Model sees Normalized).
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    
    class NormalizedModel(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m((x - mean) / std)
            
    norm_item = NormalizedModel(model)
    
    # Prepare Answer Key
    key_path = os.path.join(output_dir, "study_answer_key.csv")
    
    with open(key_path, 'w', newline='') as csvfile:
        fieldnames = ['pair_id', 'filename', 'correct_location_of_clean_image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, idx in enumerate(selected_indices):
            pair_id = i + 1
            img_path = all_files[idx]
            
            # Load & Process
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = t(img_pil).unsqueeze(0).to(DEVICE)
            
            # Generate Attack (OKLCH-PGD)
            # 0.005 is roughly 1.3/255, very subtle.
            atk_eps = SETTINGS["attack"]["oklch_eps"]
            atk_steps = SETTINGS["attack"]["steps"]
            
            with torch.no_grad():
                output = norm_item(img_tensor)
                label_tensor = output.argmax(dim=1)
            
            attacker = ChromicPGD(norm_item.m, eps_l=atk_eps, eps_c=atk_eps, eps_h=0.0, steps=atk_steps, freeze_L=False, freeze_C=False, freeze_H=True)
            img_attacked_tensor = attacker(img_tensor, label_tensor)
            
            # Convert back to PIL
            img_attacked_pil = transforms.ToPILImage()(img_attacked_tensor.squeeze().cpu())
            img_clean_pil = transforms.ToPILImage()(img_tensor.squeeze().cpu())
            
            # Make Pair
            filename, answer = create_pair(img_clean_pil, img_attacked_pil, pair_id, output_dir)
            
            print(f"Generated Pair {pair_id}: {filename} (Clean is {answer})")
            
            writer.writerow({
                'pair_id': pair_id, 
                'filename': filename, 
                'correct_location_of_clean_image': answer
            })
            
    print(f"\nSuccess! Study pack generated in '{output_dir}'")
    print(f"Answer Key: {key_path}")

if __name__ == "__main__":
    main()
