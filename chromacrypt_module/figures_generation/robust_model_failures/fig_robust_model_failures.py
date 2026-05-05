
import torch
import torch.nn as nn
import sys
from types import ModuleType
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lpips  # Ensure lpips is installed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Mock pkg_resources
try:
    import pkg_resources
except ImportError:
    mock_pkg = ModuleType("pkg_resources")
    def resource_filename(package_or_requirement, resource_name):
        return resource_name 
    mock_pkg.resource_filename = resource_filename
    sys.modules["pkg_resources"] = mock_pkg

from robustbench.utils import load_model
from torchvision import transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Running Failure Analysis on {DEVICE}...")

# -----------------------------------------------------------------------------
# Visualization Tools (Reused from generate_proofs.py)
# -----------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        print(f"DEBUG: gradients shape: {gradients.shape}, activations shape: {activations.shape}")
        
        # Handle 1D/2D gradients (Swin Transformer / Vision Transformer specifics)
        # Shape seen: (49, 1536) -> This is likely (H*W, C) for 7x7 spatial
        if len(gradients.shape) == 2:
            s_len, n_ch = gradients.shape
            if int(np.sqrt(s_len - 1)) ** 2 == (s_len - 1): # ViT CLS token check
                gradients = gradients[1:]
                activations = activations[1:]
                s_len -= 1
            side = int(np.sqrt(s_len)) 
            if side * side == s_len:
                # Reshape to (C, H, W)
                gradients = gradients.transpose(1, 0).reshape(n_ch, side, side)
                activations = activations.transpose(1, 0).reshape(n_ch, side, side)
                # print(f"DEBUG: Reshaped to {gradients.shape}")
            else:
                # Fallback, maybe just average?
                pass

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        # Vectorized weighted sum is faster and cleaner
        # cam = np.sum(weights[:, None, None] * activations, axis=0)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, class_idx

def get_saliency(model, input_tensor, class_idx=None):
    input_tensor.requires_grad_()
    input_tensor.retain_grad()
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1)
    
    score = output[0, class_idx]
    score.backward()
    
    saliency = input_tensor.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

from chromacrypt_module.attacks import NarrowbandMimicry
from chromacrypt_module.color_ops import DifferentiableColorOps



# -----------------------------------------------------------------------------
# Main Analysis
# -----------------------------------------------------------------------------

def main():
    # 1. Setup Metrics
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(DEVICE)
    loss_fn_lpips.eval()
    
    # 2. Dataset
    class FlatFolderDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            import glob
            self.files = sorted(glob.glob(os.path.join(root, "*.JPEG")))
            self.transform = transform
            
        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            path = self.files[idx]
            try:
                img = Image.open(path).convert('RGB')
                if self.transform: img = self.transform(img)
                return img, idx, os.path.basename(path)
            except: return torch.zeros(3, 224, 224), idx, "error"

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))), "data", "imagenet-1k")
    dataset = FlatFolderDataset(data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True) # Shuffle to find random pass/fails
    
    # Normalizers
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)
    normalize = lambda x: (x - mean) / std

    # Models to analyze
    models_config = [
        ("Xu2024_Swin-L", "Xu2024MIMIR_Swin-L"),
        ("Liu2023_ConvNeXt-L", "Liu2023Comprehensive_ConvNeXt-L")
    ]
    
    for model_short, model_name in models_config:
        print(f"\nAnalyzing {model_short}...")
        try:
            model = load_model(model_name=model_name, dataset='imagenet', threat_model='Linf', model_dir='../../../whitepaper_v1.3_submissions/code/models').to(DEVICE).eval()
        except Exception as e:
            print(f"Skipping {model_short} (Load Failed): {e}")
            import traceback
            traceback.print_exc()
            continue

        # Try to hook the last layer for GradCAM
        target_layer = None
        if "Swin" in model_short:
             # Swin is tricky. Usually model.layers[-1].blocks[-1].norm1 or similar
             try:
                 # Check if wrapped in Sequential/ImageNormalizer (RobustBench)
                 if hasattr(model, 'model'):
                     # model.model.layers...
                     target_layer = model.model.layers[-1].blocks[-1].norm1
                 else:
                     target_layer = model.layers[-1].blocks[-1].norm1
             except:
                 print("Could not locate Swin target layer for GradCAM. Skipping CAM.")
        elif "ConvNeXt" in model_short:
             try:
                 # ConvNeXt structure from inspection:
                 # Sequential(
                 #   (0): ImageNormalizer...
                 #   (1): ConvNeXt( ... (stages): ... )
                 # )
                 # So we need to access model[1] or model.model
                 
                 base_model = model
                 if hasattr(model, 'model'):
                     base_model = model.model
                 elif isinstance(model, nn.Sequential):
                     # Likely index 1 is the model
                     base_model = model[1]
                     
                 # Access last block of last stage
                 # Stage 3 is the last main stage. Block 2 is the last block in stage 3.
                 target_layer = base_model.stages[3].blocks[2].norm 
                 # Blocks have: conv_dw, norm, mlp. 'norm' is good for CAM/Activations before MLP?
                 # Or usually we grab the output of the block.
                 # Let's register hook on the block itself, or the norm inside it. 
                 # GradCAM on the 'norm' is safer as it's a leaf module often.
                 # Actually, usually we want the output of the CONV layers.
                 # But in Transformer/Modern ConvNets, the 'norm' before MLP is a good proxy for spatial features.
                 # Let's stick to .norm 
             except:
                 print("Could not locate ConvNeXt target layer. Skipping CAM.")
                 
        grad_cam = GradCAM(model, target_layer) if target_layer else None
        
        successes = [] # (clean, adv, clean_pred, adv_pred, lpips_val, saliency_c, saliency_a, cam_c, cam_a)
        failures = []
        
        lpips_scores = []
        
        for i, (img, idx, fname) in enumerate(loader):
            if len(successes) >= 5 and len(failures) >= 5 and len(lpips_scores) >= 100:
                break
                
            img = img.to(DEVICE)
            
            # Clean Pred
            with torch.no_grad():
                clean_logits = model(img)
                clean_pred = clean_logits.argmax(dim=1).item()
            
            # Generate Attack
            color_ops = DifferentiableColorOps().to(DEVICE)
            attacker = NarrowbandMimicry(eps=0.2, channel="C")
            img_adv = attacker(img, color_ops)
            
            # Adv Pred
            with torch.no_grad():
                adv_logits = model(img_adv)
                adv_pred = adv_logits.argmax(dim=1).item()
                
            # Compute LPIPS (Expects [-1, 1])
            # img and img_adv are [0, 1]
            l_val = loss_fn_lpips(img * 2 - 1, img_adv * 2 - 1).item()
            lpips_scores.append(l_val)
            
            is_success = (clean_pred != adv_pred)
            
            # Only store if we need more of this type
            store_success = is_success and len(successes) < 5
            store_failure = (not is_success) and len(failures) < 5
            
            if store_success or store_failure:
                # Generate Viz (Saliency / CAM)
                # Need gradients, so re-run with grad
                
                # Saliency
                sal_c = get_saliency(model, img.clone(), clean_pred)
                sal_a = get_saliency(model, img_adv.clone(), adv_pred)
                
                # CAM
                cam_c = np.zeros((224,224))
                cam_a = np.zeros((224,224))
                if grad_cam:
                    # Re-forward for hooks
                    nc = img.clone().detach().requires_grad_(True)
                    na = img_adv.clone().detach().requires_grad_(True)
                    cam_c, _ = grad_cam(nc, clean_pred)
                    cam_a, _ = grad_cam(na, adv_pred)
                
                item = {
                    'img_c': img.cpu().squeeze().permute(1,2,0).numpy(),
                    'img_a': img_adv.cpu().squeeze().permute(1,2,0).numpy(),
                    'sal_c': sal_c,
                    'sal_a': sal_a,
                    'cam_c': cam_c,
                    'cam_a': cam_a,
                    'clean_pred': clean_pred,
                    'adv_pred': adv_pred,
                    'lpips': l_val,
                    'fname': fname[0]
                }
                
                if store_success: 
                    successes.append(item)
                else: 
                    failures.append(item)
                    # Save individual failure image for user inspection
                    fail_dir = "robust_failures"
                    if not os.path.exists(fail_dir): os.makedirs(fail_dir)
                    
                    # Save Clean and Adv
                    # Adv is the one that failed to fool (Model Correct)
                    # Clean is the reference
                    Clean_name = f"{fail_dir}/{model_short}_fail_{len(failures)}_class_{clean_pred}_clean.png"
                    Adv_name = f"{fail_dir}/{model_short}_fail_{len(failures)}_class_{adv_pred}_adv.png"
                    
                    # Convert to uint8
                    Image.fromarray((item['img_c'] * 255).astype(np.uint8)).save(Clean_name)
                    Image.fromarray((item['img_a'] * 255).astype(np.uint8)).save(Adv_name)
                    
                print(f"Stored {'Success' if is_success else 'Failure'} (LPIPS={l_val:.4f}). S={len(successes)}, F={len(failures)}")

        print(f"Average LPIPS (N={len(lpips_scores)}): {sum(lpips_scores)/len(lpips_scores):.5f}")
        
        # Plotting
        def plot_grid(data_list, title, filename):
            if not data_list: return
            rows = len(data_list)
            cols = 6 # Clean, Sal, CAM | Adv, Sal, CAM
            fig, axes = plt.subplots(rows, cols, figsize=(24, 4*rows))
            if rows == 1: axes = axes.reshape(1, -1)
            
            for r, d in enumerate(data_list):
                # Clean Group
                axes[r,0].imshow(d['img_c'])
                axes[r,0].set_ylabel(f"(Class {d['clean_pred']})", rotation=90, labelpad=20, fontsize=36)
                axes[r,0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for sp in axes[r,0].spines.values(): sp.set_visible(False)
                if r == 0: axes[r,0].set_title("Clean Image", fontsize=36)
                
                axes[r,1].imshow(d['sal_c'], cmap='hot')
                if r == 0: axes[r,1].set_title("Clean\nSaliency", fontsize=36)
                axes[r,1].axis('off')
                
                axes[r,2].imshow(d['img_c'])
                axes[r,2].imshow(cv2.resize(d['cam_c'], (224,224)), alpha=0.5, cmap='jet')
                if r == 0: axes[r,2].set_title("Clean\nGrad-CAM", fontsize=36)
                axes[r,2].axis('off')
                
                # Adv Group
                axes[r,3].imshow(d['img_a'])
                if r == 0: axes[r,3].set_title(f"Attacked Image\n(Pred {d['adv_pred']})", fontsize=36)
                axes[r,3].set_xlabel(f"LPIPS: {d['lpips']:.3f}", fontsize=30)
                axes[r,3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=True)
                for sp in axes[r,3].spines.values(): sp.set_visible(False)
                
                axes[r,4].imshow(d['sal_a'], cmap='hot')
                if r == 0: axes[r,4].set_title("Attacked\nSaliency", fontsize=36)
                axes[r,4].axis('off')

                axes[r,5].imshow(d['img_a'])
                axes[r,5].imshow(cv2.resize(d['cam_a'], (224,224)), alpha=0.5, cmap='jet')
                if r == 0: axes[r,5].set_title("Attacked\nGrad-CAM", fontsize=36)
                axes[r,5].axis('off')

            plt.suptitle(title, fontsize=45)
            plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
            plt.subplots_adjust(top=0.9)
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'experiments', 'results', 'figures')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved {filename}")

        plot_grid(successes, f"{model_short} - Successful Attacks", f"fig_robust_failures_{model_short}_success.png")
        plot_grid(failures, f"{model_short} - Failed Attacks (Robust)", f"fig_robust_failures_{model_short}_fail.png")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
