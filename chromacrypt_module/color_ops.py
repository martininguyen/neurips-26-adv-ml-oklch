import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class DifferentiableBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, channels=3):
        super(DifferentiableBlur, self).__init__()
        
        # Handle Identity Case (sigma near 0)
        if sigma < 1e-5:
            self.identity = True
            return
        self.identity = False
        
        # Create a Gaussian kernel
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).reshape(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        # Calculate the 2-D gaussian kernel
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2 * variance)
                          )
        
        # Normalize so sum = 1
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # Reshape for conv2d weights: (out_channels, in_channels/groups, k, k)
        # We use groups=channels so each channel is blurred independently
        weights = gaussian_kernel.reshape(1, 1, kernel_size, kernel_size)
        self.register_buffer('weights', weights.repeat(channels, 1, 1, 1))
        
        self.padding = kernel_size // 2
        self.groups = channels

    def forward(self, x):
        if hasattr(self, 'identity') and self.identity:
            return x
        # x is (B, C, H, W) in RGB
        return F.conv2d(x, self.weights, padding=self.padding, groups=self.groups)

class DifferentiableColorOps(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Canonical M1: Linear sRGB -> LMS
        # Source: https://bottosson.github.io/posts/oklab/
        self.register_buffer('M1', torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]
        ]))
        
        # Canonical M2: LMS -> OKLab
        self.register_buffer('M2', torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660]
        ]))
        
        # Inverses
        self.register_buffer('M1_inv', torch.inverse(self.M1))
        self.register_buffer('M2_inv', torch.inverse(self.M2))

        # CIE XYZ D65 reference white
        self.register_buffer('Xn', torch.tensor(0.950489))
        self.register_buffer('Yn', torch.tensor(1.000000))
        self.register_buffer('Zn', torch.tensor(1.088840))

        # sRGB to XYZ matrix
        self.register_buffer('M_sRGB_to_XYZ', torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]))
        self.register_buffer('M_XYZ_to_sRGB', torch.inverse(self.M_sRGB_to_XYZ))

    def _flatten_spatial(self, tensor):
        orig_shape = tensor.shape
        if tensor.dim() == 4:
            tensor_flat = tensor.permute(0, 2, 3, 1).clone().reshape(-1, 3)
        elif tensor.dim() == 3:
            tensor_flat = tensor.permute(1, 2, 0).clone().reshape(-1, 3)
        else:
            tensor_flat = tensor
        return tensor_flat, orig_shape

    def _unflatten_spatial(self, tensor, orig_shape):
        if len(orig_shape) == 4:
            return tensor.reshape(orig_shape[0], orig_shape[2], orig_shape[3], 3).permute(0, 3, 1, 2).contiguous()
        elif len(orig_shape) == 3:
            return tensor.reshape(orig_shape[1], orig_shape[2], 3).permute(2, 0, 1).contiguous()
        return tensor.contiguous()

    def srgb_to_linear(self, x):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(self, x):
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * (torch.pow(torch.abs(x) + 1e-8, 1.0/2.4)) - 0.055)

    def rgb_to_oklab(self, rgb):
        # RGB (0-1, sRGB) -> Linear RGB
        lin_rgb = self.srgb_to_linear(rgb)
        
        # Reshape for matmul
        lin_rgb_flat, orig_shape = self._flatten_spatial(lin_rgb)
            
        # Linear RGB -> LMS (via M1)
        lms = torch.matmul(lin_rgb_flat, self.M1.T)
        
        # Non-linearity (Cube Root)
        lms_p = torch.sign(lms) * torch.pow(torch.abs(lms), 1.0/3.0)
        
        # LMS -> OKLab (via M2)
        oklab = torch.matmul(lms_p, self.M2.T)
        
        # Reshape back
        return self._unflatten_spatial(oklab, orig_shape)

    def oklab_to_rgb(self, oklab):
        # OKLab -> Linear RGB (inverse process)
        
        # Reshape
        oklab_flat, orig_shape = self._flatten_spatial(oklab)

        # OKLab -> LMS (via M2_inv)
        lms_p = torch.matmul(oklab_flat, self.M2_inv.T)
        
        # LMS Cube
        lms = lms_p ** 3
        
        # LMS -> Linear RGB (via M1_inv)
        lin_rgb = torch.matmul(lms, self.M1_inv.T)
        
        # Linear RGB -> sRGB
        rgb = self.linear_to_srgb(lin_rgb)
        
        # Reshape back
        return self._unflatten_spatial(rgb, orig_shape)

    def oklab_to_oklch(self, oklab):
        # oklab is (..., 3) or (..., 3, H, W)
        if oklab.dim() >= 3 and oklab.shape[-3] == 3: # channel first
            L = oklab[..., 0, :, :]
            a = oklab[..., 1, :, :]
            b = oklab[..., 2, :, :]
            dim_c = -3
        elif oklab.dim() >= 1 and oklab.shape[-1] == 3: # channel last
            L = oklab[..., 0]
            a = oklab[..., 1]
            b = oklab[..., 2]
            dim_c = -1
        else:
            raise ValueError("Unsupported shape for oklab input")
            
        C = torch.sqrt(a**2 + b**2)
        h = torch.atan2(b, a) # -pi to pi (radians)
        
        # Convert to degrees for consistency with common tools, 
        # but internal processing often uses radians.
        # Let's align with the 'poison.py' expectation which seemed to handle cyclic hue.
        # We will output DEGREES [0, 360) to match standard 'Color' object behavior
        
        h_deg = torch.rad2deg(h) % 360
        
        return torch.stack([L, C, h_deg], dim=dim_c)

    def oklch_to_oklab(self, oklch):
         if oklch.dim() >= 3 and oklch.shape[-3] == 3: # channel first
            L = oklch[..., 0, :, :]
            C = oklch[..., 1, :, :]
            h_deg = oklch[..., 2, :, :]
            dim_c = -3
         elif oklch.dim() >= 1 and oklch.shape[-1] == 3: # channel last
            L = oklch[..., 0]
            C = oklch[..., 1]
            h_deg = oklch[..., 2]
            dim_c = -1
            
         h_rad = torch.deg2rad(h_deg)
         a = C * torch.cos(h_rad)
         b = C * torch.sin(h_rad)
         
         return torch.stack([L, a, b], dim=dim_c)

    def rgb_to_oklch(self, rgb):
        return self.oklab_to_oklch(self.rgb_to_oklab(rgb))

    def oklch_to_rgb(self, oklch):
        return self.oklab_to_rgb(self.oklch_to_oklab(oklch))
        
    def f_cielab(self, t):
        delta = 6.0 / 29.0
        return torch.where(t > delta**3, torch.pow(torch.abs(t) + 1e-8, 1.0/3.0), (t / (3 * delta**2)) + (4.0 / 29.0))

    def f_inv_cielab(self, t):
        delta = 6.0 / 29.0
        return torch.where(t > delta, torch.pow(t, 3.0), 3 * delta**2 * (t - 4.0 / 29.0))

    def rgb_to_lab(self, rgb):
        lin_rgb = self.srgb_to_linear(rgb)
        
        lin_rgb_flat, orig_shape = self._flatten_spatial(lin_rgb)
            
        xyz = torch.matmul(lin_rgb_flat, self.M_sRGB_to_XYZ.T)
        
        x = xyz[:, 0] / self.Xn
        y = xyz[:, 1] / self.Yn
        z = xyz[:, 2] / self.Zn
        
        fx = self.f_cielab(x)
        fy = self.f_cielab(y)
        fz = self.f_cielab(z)
        
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        
        lab = torch.stack([L, a, b], dim=-1)
        
        return self._unflatten_spatial(lab, orig_shape)

    def lab_to_rgb(self, lab):
        lab_flat, orig_shape = self._flatten_spatial(lab)
            
        L = lab_flat[:, 0]
        a = lab_flat[:, 1]
        b = lab_flat[:, 2]
        
        fy = (L + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0
        
        x = self.f_inv_cielab(fx) * self.Xn
        y = self.f_inv_cielab(fy) * self.Yn
        z = self.f_inv_cielab(fz) * self.Zn
        
        xyz = torch.stack([x, y, z], dim=-1)
        
        lin_rgb = torch.matmul(xyz, self.M_XYZ_to_sRGB.T)
        rgb = self.linear_to_srgb(lin_rgb)
        
        return self._unflatten_spatial(rgb, orig_shape)
        
    def gamut_clip_preserve_hue(self, oklch, steps=12):
        """
        Differentiable approximation of gamut mapping by finding max valid Chroma.
        We binary search the Chroma C along the line [0, C] to find the boundary.
        Preserves L and H exactly.
        """
        # Determine Check Dims
        if oklch.dim() >= 3 and oklch.shape[-3] == 3:
            c_idx = 1
            # Extract L, H for reconstruction logic
            # L = oklch[:, 0:1, :, :]
            # H = oklch[:, 2:3, :, :]
            C_orig = oklch[:, 1:2, :, :]
        elif oklch.dim() >= 1 and oklch.shape[-1] == 3:
            c_idx = -1
            # L = oklch[..., 0:1]
            # H = oklch[..., 2:3]
            C_orig = oklch[..., 1:2]
        else:
            return oklch
            
        low = torch.zeros_like(C_orig)
        high = C_orig
        
        # 10 steps of binary search is usually sufficient for RGB 8-bit precision
        # (1/2^10 ~= 0.001)
        # We want to find the MAX C that is valid.
        
        # Helper to check validity
        def is_valid(C_test):
            # Construct OKLCH candidate
            if c_idx == 1:
                cand = torch.cat([oklch[:, 0:1, :, :], C_test, oklch[:, 2:3, :, :]], dim=1)
            else:
                cand = torch.cat([oklch[..., 0:1], C_test, oklch[..., 2:3]], dim=-1)
            
            rgb = self.oklch_to_rgb(cand)
            
            # Epsilon tolerance for "soft" validity
            eps = 1e-4
            mask_lower = (rgb < -eps).any(dim=c_idx if c_idx == -1 else -3, keepdim=True)
            mask_upper = (rgb > 1 + eps).any(dim=c_idx if c_idx == -1 else -3, keepdim=True)
            return ~(mask_lower | mask_upper)
            
        # Binary Search Loop
        # We always maintain: 'low' is valid (or 0), 'high' attempts to reach C_orig
        
        # Optimization: First check if original high is valid
        # This saves compute for the 99% of pixels that are already valid
        mask_valid_high = is_valid(high)
        # If valid, we are done for those pixels?
        # We can implement this optimization by masked updates, or just run the search everywhere.
        # Running everywhere is simpler for differentiability logic, but slightly slower.
        # Let's run everywhere for robustness.
        
        for _ in range(steps): # Number of iterations for precision
            mid = (low + high) * 0.5
            valid_mask = is_valid(mid)
            
            # If mid is valid, we can try higher C (so low = mid)
            # If mid is invalid, we must go lower (so high = mid)
            low = torch.where(valid_mask, mid, low)
            high = torch.where(valid_mask, high, mid)
            
        # Construct final Valid OKLCH
        # Use 'low' because it guarantees validity (as much as possible)
        # 'high' might be just outside.
        if c_idx == 1:
            final_oklch = torch.cat([oklch[:, 0:1, :, :], low, oklch[:, 2:3, :, :]], dim=1)
        else:
            final_oklch = torch.cat([oklch[..., 0:1], low, oklch[..., 2:3]], dim=-1)

        return final_oklch
