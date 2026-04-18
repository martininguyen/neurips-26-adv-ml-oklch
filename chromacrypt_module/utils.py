import os
import sys
import glob
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from chromacrypt_module.color_ops import DifferentiableColorOps
import warnings

# --- GLOBAL SECURITY ENFORCEMENT & TELEMETRY ---
# Intercepts nested arbitrary execution vectors during checkpoint unpacking globally
_native_torch_load_ptr = torch.load
def _secure_weight_loader(*args, **kwargs):
    kwargs['weights_only'] = True
    return _native_torch_load_ptr(*args, **kwargs)
torch.load = _secure_weight_loader

# Suppress generic TorchVision V1 deprecation warnings (LPIPS explicitly requires V1 weight dimensionality) 
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')
warnings.filterwarnings('ignore', category=FutureWarning)
# -----------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMAGENET_SEARCH_PATHS = [
    os.path.join(_PROJECT_ROOT, "data", "imagenet-1k"),
    os.path.join(_PROJECT_ROOT, "data", "datasets", "imagenet-1k"),
]

def load_env():
    # Crawl upwards natively parsing .env to avoid missing pip package constraints
    current_dir = os.path.abspath(__file__)
    for _ in range(5):
        current_dir = os.path.dirname(current_dir)
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip("'").strip('"')
            if "HUGGINGFACE_ACCESS_TOKEN" in os.environ:
                # Force override to prevent Windows system env conflicts
                os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
            import huggingface_hub
            huggingface_hub.login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
            return

# Auto-execute env logic to instantly satisfy down-stream module initializations natively
load_env()

def find_imagenet_dir():
    for d in _IMAGENET_SEARCH_PATHS:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"ImageNet-1K not found in standard directories.")

def load_victim_model():
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE).eval()
    except Exception:
        model = models.resnet50(pretrained=True).to(DEVICE).eval()
    return model

def load_imagenet_val_batch(n_examples=20, offset=0):
    data_dir = find_imagenet_dir()
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.JPEG")) + glob.glob(os.path.join(data_dir, "*.jpg")))
    
    selected = image_paths[offset:offset + n_examples]
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    batch = torch.stack([t(Image.open(p).convert("RGB")) for p in selected]).to(DEVICE)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    
    label_file = os.path.join(data_dir, "val_pytorch_labels.txt")
    if os.path.exists(label_file):
        with open(label_file) as f:
            all_labels = [int(line.strip()) for line in f.readlines()]
        labels = torch.tensor(all_labels[offset:offset + n_examples], dtype=torch.long).to(DEVICE)
    else:
        model = load_victim_model()
        with torch.no_grad():
            labels = model((batch - mean) / std).argmax(1)
            
    return batch, labels, mean, std

# ---------------------------------------------------------------------------
# Model Wrappers (for gradient-based optimization in perceptual color spaces)
# ---------------------------------------------------------------------------

class OKLCHModelWrapper(nn.Module):
    """
    Wraps a standard RGB classifier to accept NORMALIZED OKLCH inputs [0, 1].
    This allows torchattacks.PGD to optimize using standard L-inf box projection
    without breaking gradients through modular hue arithmetic.

    Normalization: L ∈ [0,1], C ∈ [0, 0.4] → [0,1], H ∈ [0, 360] → [0,1]
    """
    def __init__(self, base_model, freeze_L=False, freeze_C=False, freeze_H=False):
        super().__init__()
        self.base_model = base_model
        self.color_ops = DifferentiableColorOps().to(DEVICE)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE))

        self.freeze_L = freeze_L
        self.freeze_C = freeze_C
        self.freeze_H = freeze_H

        # Injected before optimization: the clean OKLCH image in normalized form
        self.clean_oklch_norm = None

    def unscale(self, x_norm):
        """Re-scales normalized variables to true OKLCH bounds."""
        L = x_norm[:, 0:1, :, :]
        C = x_norm[:, 1:2, :, :] * 0.4
        H = x_norm[:, 2:3, :, :] * 360.0

        # Freeze channels by substituting clean values
        if self.clean_oklch_norm is not None:
            clean_norm = self.clean_oklch_norm
            
            # Handle batch shrinking (e.g., from torchattacks.Square)
            if x_norm.shape[0] != clean_norm.shape[0]:
                L_flat_x = x_norm[:, 0].view(x_norm.shape[0], -1)
                L_flat_clean = clean_norm[:, 0].view(clean_norm.shape[0], -1)
                diff = (L_flat_x.unsqueeze(1) - L_flat_clean.unsqueeze(0)).abs().sum(dim=2)
                matched_indices = diff.argmin(dim=1)
                clean_norm = clean_norm[matched_indices]

            if self.freeze_L: L = clean_norm[:, 0:1, :, :]
            if self.freeze_C: C = clean_norm[:, 1:2, :, :] * 0.4
            if self.freeze_H: H = clean_norm[:, 2:3, :, :] * 360.0

        return torch.cat([L, C, H], dim=1)

    def forward(self, x_norm):
        oklch_input = self.unscale(x_norm)
        clipped_oklch = self.color_ops.gamut_clip_preserve_hue(oklch_input, steps=12)
        rgb_out = self.color_ops.oklch_to_rgb(clipped_oklch).contiguous().clamp(0.0, 1.0)
        rgb_norm_tensor = (rgb_out - self.mean) / self.std
        return self.base_model(rgb_norm_tensor)


class CIELABModelWrapper(nn.Module):
    """
    Wraps a standard RGB classifier to accept NORMALIZED CIELAB inputs [0, 1].

    Normalization: L ∈ [0,100] → [0,1], a ∈ [-128,127] → [0,1], b ∈ [-128,127] → [0,1]
    """
    def __init__(self, base_model, freeze_L=False, freeze_A=False, freeze_B=False):
        super().__init__()
        self.base_model = base_model

        device = next(base_model.parameters()).device
        self.color_ops = DifferentiableColorOps().to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

        self.freeze_L = freeze_L
        self.freeze_A = freeze_A
        self.freeze_B = freeze_B
        self.clean_lab_norm = None

    def unscale(self, x_norm):
        """Re-scales normalized variables to true CIELAB bounds."""
        L = x_norm[:, 0:1, :, :] * 100.0
        A = (x_norm[:, 1:2, :, :] * 255.0) - 128.0
        B = (x_norm[:, 2:3, :, :] * 255.0) - 128.0

        if self.clean_lab_norm is not None:
            clean_norm = self.clean_lab_norm
            
            if x_norm.shape[0] != clean_norm.shape[0]:
                L_flat_x = x_norm[:, 0].view(x_norm.shape[0], -1)
                L_flat_clean = clean_norm[:, 0].view(clean_norm.shape[0], -1)
                diff = (L_flat_x.unsqueeze(1) - L_flat_clean.unsqueeze(0)).abs().sum(dim=2)
                matched_indices = diff.argmin(dim=1)
                clean_norm = clean_norm[matched_indices]

            if self.freeze_L: L = clean_norm[:, 0:1, :, :] * 100.0
            if self.freeze_A: A = (clean_norm[:, 1:2, :, :] * 255.0) - 128.0
            if self.freeze_B: B = (clean_norm[:, 2:3, :, :] * 255.0) - 128.0

        return torch.cat([L, A, B], dim=1)

    def forward(self, x_norm):
        lab_input = self.unscale(x_norm)
        rgb_out = self.color_ops.lab_to_rgb(lab_input).contiguous()
        rgb_norm_tensor = (rgb_out - self.mean) / self.std
        return self.base_model(rgb_norm_tensor)
