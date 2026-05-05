"""
Summary: Generates structured adversarial image assets tailored for CAPTCHA human-subject studies.
Objective: Demonstrates that humans can successfully solve OKLCH-perturbed visual tasks (e.g., identifying vehicles or animals) even when SOTA machine vision fails completely.
Execution: Samples images from designated pools, applies parameterized OKLCH grid attacks via JSON configs, and exports the isolated test arrays.
"""


import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import glob
import random
import csv
import argparse
import numpy as np
import json

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "experiment_captcha_study_gen.json"), "r") as f:
    SETTINGS = json.load(f)

# Ensure we can import the color ops
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "working_scripts"))
    sys.path.append(os.path.join(project_root, "oklch_defense"))
try:
    from chromacrypt_module.color_ops import DifferentiableColorOps
    from chromacrypt_module.attacks import NarrowbandMimicry
except ImportError:
    print("Error: chromacrypt_module not found.")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Removed inline apply_narrowband_noise and apply_structural_attack in favor of canonical NarrowbandMimicry

def main():
    output_dir = SETTINGS["dataset"]["output_dir"]
    pool_targets = SETTINGS["generation"]["pool_targets"]
    total_grids = SETTINGS["generation"]["total_grids"]
    
    # Calculate num_samples implicitly from pool_targets for printing
    num_samples = sum(pool_targets.values())
    
    # Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {num_samples} CAPTCHA source images in '{output_dir}'...")
    
    # Load Model (Needed for Gradient Calculation & Labels)
    print("Loading ResNet50 for Gradient and Labels...")
    import torchvision.models as models
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(DEVICE).eval()
    categories = weights.meta["categories"]
    
    color_ops = DifferentiableColorOps().to(DEVICE)
    
    # Define Simplified Class Logic
    def get_simplified_label(label):
        label = label.lower()
        
        # Mapping Rules (Order matters for priority)
        mappings = {
            "dog": ["dog", "terrier", "retriever", "hound", "spaniel", "collie", "poodle", "corgi", "husky", "vizsla", "dalmatian", "beagle", "shepherd", "rottweiler", "boxer", "bulldog", "pug", "chihuahua", "setter", "pointer", "pinscher", "schnauzer", "dane", "mastiff", "sheepdog", "maltese", "papillon", "pekinese", "shih-tzu", "blenheim", "borzoi", "basenji", "kelpie", "malinois", "schipperke", "groenendael", "briard", "komondor", "kuvasz", "affenpinscher", "lhasa", "keeshond", "chow", "pomeranian", "samoyed"],
            "cat": ["cat", "tabby", "tiger", "lion", "leopard", "jaguar", "cheetah", "cougar", "lynx", "siamese", "persian", "burmese", "egyptian"],
            "bird": ["bird", "jay", "eagle", "owl", "hawk", "robin", "hummingbird", "penguin", "swan", "goose", "duck", "chicken", "rooster", "hen", "ostrich", "flamingo", "parrot", "vulture", "sparrow", "finch", "crane", "stork", "pelican", "seagull", "albatross", "kite", "ptarmigan", "grouse", "peacock", "quail", "partridge", "lorikeet", "macaw", "cockatoo", "toucan", "hornbill", "kingfisher", "magpie", "chickadee", "bulbul", "goldfinch", "brambling"],
            "vehicle": ["car", "cab", "taxi", "bus", "truck", "ambulance", "racer", "minivan", "jeep", "limo", "scooter", "moped", "bicycle", "bike", "motorcycle", "train", "airplane", "aeroplane", "jet", "ship", "boat", "canoe", "kayak", "raft", "sailboat", "liner", "yacht", "trolley", "tram", "tractor", "harvester", "plow", "trailer", "van", "wagon", "convertible"],
            "fruit": ["apple", "orange", "banana", "lemon", "pineapple", "strawberry", "grape", "pear", "peach", "cherry", "fig", "pomegranate"],
            "insect": ["ant", "bee", "wasp", "beetle", "butterfly", "moth", "fly", "mosquito", "grasshopper", "cricket", "mantis", "dragonfly", "spider", "tarantula", "scorpion", "weevil", "ladybug", "cicada", "leafhopper"],
            "aquatic": ["fish", "shark", "whale", "dolphin", "crab", "lobster", "shrimp", "starfish", "jellyfish", "squid", "octopus", "ray", "barracuda", "goldfish", "eel", "salmon", "trout", "bass", "pike", "sturgeon", "gar"],
            "fungus": ["mushroom", "toadstool", "fungus", "gyromitra", "morel", "agaric", "bolete", "stinkhorn", "truffle"],
            "food": ["bagel", "pizza", "burger", "bread", "cheese", "cake", "pie", "cookie", "chocolate", "ice cream", "sandwich", "burrito", "taco", "hotdog", "pretzel", "dough", "meatloaf", "guacamole", "consomme", "trifle"],
            "instrument": ["guitar", "piano", "violin", "cello", "flute", "trumpet", "drum", "saxophone", "sax", "harp", "banjo", "accordion", "harmonica", "trombone", "clarinet", "oboe", "bassoon", "marimba", "xylophone", "chime", "gong", "maraca"]
        }
        
        for simple_cat, keywords in mappings.items():
            for kw in keywords:
                if kw in label:
                    return simple_cat
        return None

    # Load ImageNet images
    img_dir = os.path.join(project_root, "data", "imagenet-1k")
        
    all_files = sorted(glob.glob(os.path.join(img_dir, "*.JPEG")))
    if not all_files:
        print("Error: No ImageNet images found.")
        return

    # Randomly shuffle for diversity
    random.shuffle(all_files)
    
    t = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor()
    ])
    
    # Classification Transform
    t_cls = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare Answer Key
    key_path = os.path.join(output_dir, "captcha_grid_key.csv")
    
    pool_counts = {"animal": 0, "vehicle": 0, "food": 0}
    
    # Storage for the pool
    # format: { "animal": [ (img_pil, filename), ... ], ... }
    image_pool = {"animal": [], "vehicle": [], "food": []}
    
    # Mapping Super-Categories
    # We need to map the "simple_label" (e.g. dog) to our 3 super-categories
    # EXCLUDING fungus from food.
    # EXCLUDING insect from animal (user request: "insects are not animals").
    super_map = {
        "dog": "animal", "cat": "animal", "bird": "animal", "aquatic": "animal",
        "vehicle": "vehicle",
        "food": "food", "fruit": "food"
    }
    
    # Mapping Super-Categories
    # We need to map the "simple_label" (e.g. dog) to our 3 super-categories
    # EXCLUDING fungus from food.
    # EXCLUDING insect from animal (user request: "insects are not animals").
    super_map = {
        "dog": "animal", "cat": "animal", "bird": "animal", "aquatic": "animal",
        "vehicle": "vehicle",
        "food": "food", "fruit": "food"
    }
    
    print(f"Generating Image Pool ({pool_targets})...")
    
    total_pool_size = sum(pool_targets.values())
    current_pool_size = 0
    
    for img_path in all_files:
        if current_pool_size >= total_pool_size:
            break
            
        # Load & Process
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except:
            continue
            
        # Get Label
        img_cls_tensor = t_cls(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_idx = model(img_cls_tensor).argmax(dim=1).item()
            raw_label = categories[pred_idx]
        
        simple_label = get_simplified_label(raw_label)
        if not simple_label or simple_label not in super_map:
            continue 
            
        super_cat = super_map[simple_label]
        
        if pool_counts[super_cat] >= pool_targets[super_cat]:
            continue
        
        # Add to Pool (Apply Attack NOW to save computation)
        # Randomize Amp for variety in the pool
        amp_choices = SETTINGS["attack"]["amplitudes"]
        amp = random.choice(amp_choices)
        
        img_tensor = t(img_pil).unsqueeze(0).to(DEVICE)
        freq_mult = SETTINGS["attack"]["freq_mult"]
        attacker = NarrowbandMimicry(eps=amp, freq_mult=freq_mult, channel="L")
        img_adv_tensor = attacker(img_tensor, color_ops)
        img_adv_pil = transforms.ToPILImage()(img_adv_tensor.squeeze().cpu())
        
        pool_counts[super_cat] += 1
        current_pool_size += 1
        
        original_filename = os.path.basename(img_path)
        image_pool[super_cat].append({
            "image": img_adv_pil,
            "label": simple_label,
            "super_category": super_cat,
            "original_filename": original_filename
        })
        
        print(f"Pooled ({current_pool_size}/{total_pool_size}): {super_cat} - {simple_label}")

    if current_pool_size < total_pool_size:
        print("Error: Could not fill image pool. Check filters.")
        return

    # --- Generate 3x3 Grids ---
    print(f"\nGenerating {total_grids} (3x3) CAPTCHA Grids...")
    
    # Grid Settings
    grid_size = (3, 3)
    cell_size = 512 // 2 # Resize 512px images to 256px for grid (Total 768x768)
    
    # Prompts to rotate through
    prompts = ["animal", "vehicle", "food"]
    
    with open(key_path, 'w', newline='') as csvfile:
        fieldnames = ['grid_id', 'prompt', 'target_cells', 'grid_filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for g_idx in range(total_grids):
            grid_id = f"G-{g_idx+1:02d}"
            
            # Select Prompt
            target_cat = prompts[g_idx % len(prompts)]
            prompt_text = f"Select all {target_cat}s"
            
            # Select 3 to 5 targets
            num_targets = random.randint(3, 5)
            
            # Select Targets (Unique)
            targets = random.sample(image_pool[target_cat], num_targets)
            
            # Select Distractors (Unique)
            other_cats = [c for c in prompts if c != target_cat]
            all_distractors = []
            for cat in other_cats:
                all_distractors.extend(image_pool[cat])
            
            distractors = random.sample(all_distractors, 9 - num_targets)
                
            # Combine and Shuffle
            cells = targets + distractors
            grid_items = []
            for item in cells:
                # Identify if this specific item is from the target category
                is_target = (item["super_category"] == target_cat)
                grid_items.append({"item": item, "is_target": is_target})
                
            random.shuffle(grid_items)
            
            # Build Image
            # Canvas size: 3 cols * 256, 3 rows * 256
            canvas = Image.new('RGB', (cell_size * 3, cell_size * 3), (255, 255, 255))
            
            target_indices = []
            
            for i, cell_data in enumerate(grid_items):
                row = i // 3
                col = i % 3
                
                img = cell_data["item"]["image"].resize((cell_size, cell_size))
                canvas.paste(img, (col * cell_size, row * cell_size))
                
                if cell_data["is_target"]:
                    target_indices.append(i) # 0-8 index
            
            # Save Grid
            filename = f"captcha_grid_{grid_id}_{target_cat}.png"
            canvas.save(os.path.join(output_dir, filename))
            
            # Format indices as string "0 3 4"
            target_str = " ".join(map(str, sorted(target_indices)))
            
            print(f"Generated {grid_id}: {prompt_text} -> Cells {target_str}")
            
            writer.writerow({
                'grid_id': grid_id,
                'prompt': prompt_text,
                'target_cells': target_str,
                'grid_filename': filename
            })

    print(f"\nSuccess! Generated {total_grids} CAPTCHA Grids in '{output_dir}'")
    print(f"Grid Key: {key_path}")

if __name__ == "__main__":
    main()
