import torch
import numpy as np
import torch.nn as nn
from .color_ops import DifferentiableColorOps

class ChromicPGD:
    def __init__(self, model, eps_l=0.01, eps_c=0.01, eps_h=0.0, steps=10, freeze_L=False, freeze_C=False, freeze_H=True):
        self.device = next(model.parameters()).device
        self.model = model
        self.color_ops = DifferentiableColorOps().to(self.device)
        self.eps_l = eps_l
        self.eps_c = eps_c
        self.eps_h = eps_h
        self.steps = steps
        
        self.alpha_l = eps_l / 4.0 if eps_l > 0 else 0
        self.alpha_c = eps_c / 4.0 if eps_c > 0 else 0
        self.alpha_h = eps_h / 4.0 if eps_h > 0 else 0
        
        self.freeze_L = freeze_L
        self.freeze_C = freeze_C
        self.freeze_H = freeze_H
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, images, labels):
        oklch_base = self.color_ops.rgb_to_oklch(images).detach()
        adv_oklch = oklch_base.clone().detach()
        
        if not self.freeze_L and self.eps_l > 0:
            adv_oklch[:, 0, :, :] += torch.empty_like(adv_oklch[:, 0, :, :]).uniform_(-self.eps_l, self.eps_l)
        if not self.freeze_C and self.eps_c > 0:
            adv_oklch[:, 1, :, :] += torch.empty_like(adv_oklch[:, 1, :, :]).uniform_(-self.eps_c, self.eps_c)
        if not self.freeze_H and self.eps_h > 0:
            adv_oklch[:, 2, :, :] += torch.empty_like(adv_oklch[:, 2, :, :]).uniform_(-self.eps_h, self.eps_h)

        for _ in range(self.steps):
            adv_oklch = adv_oklch.contiguous()
            adv_oklch.requires_grad = True
            
            if adv_oklch.grad is not None:
                adv_oklch.grad.data.zero_()
            
            adv_rgb = self.color_ops.oklch_to_rgb(self.color_ops.gamut_clip_preserve_hue(adv_oklch, steps=12)).contiguous().clamp(0, 1)
            outputs = self.model((adv_rgb - self.mean) / self.std)
            cost = self.loss_fn(outputs, labels)
            
            grad = torch.autograd.grad(cost, adv_oklch, retain_graph=False, create_graph=False)[0]
            
            if self.freeze_L:
                grad[:, 0, :, :] = 0 
            if self.freeze_C:
                grad[:, 1, :, :] = 0
            if self.freeze_H:
                grad[:, 2, :, :] = 0
            
            adv_oklch = adv_oklch.detach()
            adv_oklch[:, 0, :, :] += self.alpha_l * grad[:, 0, :, :].sign()
            adv_oklch[:, 1, :, :] += self.alpha_c * grad[:, 1, :, :].sign()
            adv_oklch[:, 2, :, :] += self.alpha_h * grad[:, 2, :, :].sign()
            
            delta = adv_oklch - oklch_base
            
            delta[:, 0, :, :] = torch.clamp(delta[:, 0, :, :], -self.eps_l, self.eps_l) if not self.freeze_L else 0
            delta[:, 1, :, :] = torch.clamp(delta[:, 1, :, :], -self.eps_c, self.eps_c) if not self.freeze_C else 0
            delta[:, 2, :, :] = torch.clamp(delta[:, 2, :, :], -self.eps_h, self.eps_h) if not self.freeze_H else 0
            
            adv_oklch = oklch_base + delta
            
            adv_oklch[:, 0, :, :] = torch.clamp(adv_oklch[:, 0, :, :], 0.0, 1.0)
            adv_oklch[:, 1, :, :] = torch.clamp(adv_oklch[:, 1, :, :], 0.0, 0.4)
            adv_oklch[:, 2, :, :] = adv_oklch[:, 2, :, :] % 360.0
            
        return self.color_ops.oklch_to_rgb(self.color_ops.gamut_clip_preserve_hue(adv_oklch.contiguous(), steps=12)).clamp(0, 1).detach()

class LuminancePGD(ChromicPGD):
    def __init__(self, model, eps_l=0.01, steps=10):
        super().__init__(model, eps_l=eps_l, eps_c=0.0, eps_h=0.0, steps=steps, freeze_L=False, freeze_C=True, freeze_H=True)

class RGBPGD:
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10):
        self.device = next(model.parameters()).device
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, images, labels):
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0, 1)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            outputs = self.model((adv_images - self.mean) / self.std)
            cost = self.loss_fn(outputs, labels)
            
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
            
        return adv_images

class NarrowbandMimicry:
    def __init__(self, eps=0.2, freq_mult=2.0, bw=2.0):
        self.eps = eps
        self.freq_mult = freq_mult
        self.bw = bw

    def __call__(self, images, color_ops):
        b, c, h, w = images.shape
        device = images.device
        noise = generate_narrowband_noise(h, w, device, self.freq_mult, self.bw)
        
        img_oklch = color_ops.rgb_to_oklch(images)
        L = img_oklch[:, 0:1, :, :]
        
        L_adv = (L + self.eps * noise).clamp(0, 1)
        adv_oklch = torch.cat([L_adv, img_oklch[:, 1:2, :, :], img_oklch[:, 2:3, :, :]], dim=1)
        
        return color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(adv_oklch, steps=12)).clamp(0, 1).detach()


class TopologicalAttractor:
    def __init__(self, eps=0.2):
        self.eps = eps

    def __call__(self, images, color_ops):
        b, c, h, w = images.shape
        device = images.device
        
        grid_L = generate_topological_grid(h, w, device)
        grid_C = grid_L  # Identical spatial geometry
        
        img_oklch = color_ops.rgb_to_oklch(images)
        L = img_oklch[:, 0:1, :, :]
        C = img_oklch[:, 1:2, :, :]
        
        L_adv = (L + self.eps * grid_L).clamp(0, 1)
        C_adv = (C + self.eps * 0.4 * grid_C).clamp(0, 0.4)
        
        adv_oklch = torch.cat([L_adv, C_adv, img_oklch[:, 2:3, :, :]], dim=1)
        
        return color_ops.oklch_to_rgb(color_ops.gamut_clip_preserve_hue(adv_oklch, steps=12)).clamp(0, 1).detach()
class AdvPatch:
    def __init__(self, model=None, patch_size=32, steps=40, alpha=0.05):
        self.model = model
        self.patch_size = patch_size
        self.steps = steps
        self.alpha = alpha
        self.loss_fn = nn.CrossEntropyLoss()
        
        if model is not None:
            self.device = next(model.parameters()).device
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def __call__(self, images, labels=None, color_ops=None):
        if self.model is None or labels is None:
            # Fallback to random noise if no model/labels provided for optimization
            b, c, h, w = images.shape
            cy, cx = h // 2, w // 2
            y1, y2 = max(0, cy - self.patch_size//2), min(h, cy + self.patch_size//2)
            x1, x2 = max(0, cx - self.patch_size//2), min(w, cx + self.patch_size//2)
            adv_img = images.clone()
            noise = torch.rand(b, c, y2-y1, x2-x1, device=images.device)
            adv_img[:, :, y1:y2, x1:x2] = noise
            return adv_img

        b, c, h, w = images.shape
        cy, cx = h // 2, w // 2
        y1, y2 = max(0, cy - self.patch_size//2), min(h, cy + self.patch_size//2)
        x1, x2 = max(0, cx - self.patch_size//2), min(w, cx + self.patch_size//2)
        
        mask = torch.zeros_like(images).to(images.device)
        mask[:, :, y1:y2, x1:x2] = 1.0
        
        adv_images = images.clone().detach()
        random_patch = torch.rand((b, c, y2-y1, x2-x1), device=images.device)
        adv_images[:, :, y1:y2, x1:x2] = random_patch
        
        adv_images.requires_grad = True
        
        for _ in range(self.steps):
            norm_imgs = (adv_images - self.mean) / self.std
            logits = self.model(norm_imgs)
            loss = self.loss_fn(logits, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad
            if grad is None: break
            
            with torch.no_grad():
                adv_images.data += mask * self.alpha * grad.sign()
                adv_images.data = torch.clamp(adv_images.data, 0, 1)
            
            adv_images.grad.zero_()
            
        return adv_images.detach()

def generate_topological_grid(h, w, device):
    """Canonical DCT-aligned checkerboard grid (Chromic Interference)"""
    x = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)
    y = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
    return torch.cos(x / 2.0) * torch.cos(y / 2.0)

def generate_narrowband_noise(h, w, device, freq_mult=2.0, bw=2.0):
    """Canonical Narrowband Noise (for Mimicry analysis)"""
    base_freq = 224.0 / (4 * np.pi) 
    target_k = base_freq * freq_mult
    
    fx = torch.fft.fftfreq(w, d=1.0).to(device) * w
    fy = torch.fft.fftfreq(h, d=1.0).to(device) * h
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    
    sigma = bw * freq_mult
    
    dist1 = (FX - target_k)**2 + (FY - target_k)**2
    dist2 = (FX + target_k)**2 + (FY + target_k)**2
    dist3 = (FX - target_k)**2 + (FY + target_k)**2
    dist4 = (FX + target_k)**2 + (FY - target_k)**2
    
    mask = torch.exp(-dist1/(2*sigma**2)) + \
           torch.exp(-dist2/(2*sigma**2)) + \
           torch.exp(-dist3/(2*sigma**2)) + \
           torch.exp(-dist4/(2*sigma**2))
           
    real_noise = torch.randn_like(mask)
    imag_noise = torch.randn_like(mask)
    filtered_f = (real_noise + 1j * imag_noise) * mask
    grid_noise = torch.fft.ifft2(filtered_f).real
    
    grid_noise = grid_noise / (grid_noise.std() + 1e-8) * 0.707
    return grid_noise.view(1, 1, h, w)
