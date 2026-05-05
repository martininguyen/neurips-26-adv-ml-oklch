import torch
import numpy as np

def generate_narrowband_noise(b, h, w, device, freq_mult=1.0, bw=2.0):
    base_freq = 224.0 / 16.0 
    target_k = base_freq * freq_mult
    
    fx = torch.fft.fftfreq(w, d=1.0).to(device) * w
    fy = torch.fft.fftfreq(h, d=1.0).to(device) * h
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    
    sigma = bw * freq_mult
    
    rad_dist = torch.sqrt(FX**2 + FY**2)
    mask = torch.exp(-((rad_dist - target_k)**2) / (2 * sigma**2))
           
    mask = mask.unsqueeze(0).expand(b, h, w)
    real_noise = torch.randn_like(mask)
    imag_noise = torch.randn_like(mask)
    filtered_f = (real_noise + 1j * imag_noise) * mask
    grid_noise = torch.fft.ifft2(filtered_f).real
    
    grid_noise_flat = grid_noise.view(b, -1)
    max_vals = grid_noise_flat.abs().max(dim=1, keepdim=True)[0] + 1e-8
    grid_noise_flat = grid_noise_flat / max_vals
    
    print("max vals:", max_vals.flatten().tolist())
    return grid_noise_flat.view(b, 1, h, w)

if __name__ == "__main__":
    out = generate_narrowband_noise(2, 224, 224, "cpu")
    print("final noise shape:", out.shape)
    print("final noise min/max (b=0):", out[0].min().item(), out[0].max().item())
    print("final noise min/max (b=1):", out[1].min().item(), out[1].max().item())
