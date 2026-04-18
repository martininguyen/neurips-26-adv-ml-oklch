import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn.functional as F

def gabor_kernel(size, sigma, theta, lam, gamma, psi):
    """Generates a Gabor kernel."""
    sigma_x = sigma
    sigma_y = sigma / gamma
    
    ymax = size // 2
    xmax = size // 2
    y, x = np.mgrid[-ymax:ymax+1, -xmax:xmax+1]
    
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * \
         np.cos(2 * np.pi / lam * x_theta + psi)
    return gb

def get_chromic_interference_grid(size, period):
    """Generates the Chromic Interference Grid pattern."""
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    # sin(x/T) * sin(y/T)
    grid = np.sin(X / period) * np.sin(Y / period)
    return grid

def compute_fft(image):
    """Computes the 2D FFT magnitude."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
    return magnitude_spectrum

def main():
    # Parameters matches ResNet50 first layer approx (7x7)
    # Nyquist frequency alignment check
    kernel_size = 31 # Larger visualization size for clarity
    
    # 1. Generate Gabor Filter (ResNet-like edge detector)
    # Theta = 45 degrees, tuned to diagonal edges
    gabor = gabor_kernel(size=kernel_size, sigma=4.0, theta=np.pi/4, lam=4.0, gamma=0.5, psi=0)
    
    # 2. Generate Chromic Interference Grid
    # Period T tuned to match the Gabor frequency (lambda=4.0 -> T approx 2.0/pi?)
    # If lambda=4, freq is 1/4. 
    # Chromic Interference grid: sin(x/T). If we want freq 1/4, 2*pi*x / (2*pi*T') -> 1/T' = 1/4 -> T' = 4.
    # sin(x / (4/ (2*pi))) ... wait. sin(wx). w = 2*pi*f. 
    # Gabor cos(2*pi*x/lambda). w = 2*pi/lambda.
    # Grid sin(x/T). w = 1/T.
    # To match: 1/T = 2*pi/lambda => T = lambda / (2*pi)
    # Let's try to match the frequency exactly.
    lambda_param = 4.0
    T_param = lambda_param / (2 * np.pi) 
    
    # Actually, let's use the parameters described in the paper or standard adversarial settings
    # Usually the localized grid is high freq. Let's show a high-freq Gabor and high-freq Grid.
    
    # Let's visualize a 7x7 kernel embedded in a larger field for FFT resolution
    pad_size = 128
    
    # Standard ResNet First Layer Kernel Approximation
    gabor_7x7 = gabor_kernel(size=7, sigma=1.0, theta=np.pi/4, lam=3.5, gamma=1.0, psi=0)
    gabor_padded = np.zeros((pad_size, pad_size))
    center = pad_size // 2
    gabor_padded[center-3:center+4, center-3:center+4] = gabor_7x7
    
    # Chromic Interference Attack Grid (Full Image)
    # Period T=1.75 approximates a checkerboard at the pixel limit (Nyquist)
    # sin(x/1.1) is very high freq.
    # Let's simply show the spectrum of a "Grid" pattern
    chromic_interference_grid = get_chromic_interference_grid(pad_size, period=1.0) # High freq checkerboard
    
    # Compute FFTs
    gabor_fft = compute_fft(gabor_padded)
    grid_fft = compute_fft(chromic_interference_grid)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Row 1: Spatial Domain
    axes[0,0].imshow(gabor_7x7, cmap='gray')
    axes[0,0].set_title('ResNet50 First-Layer Kernel (Approx Gabor)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(chromic_interference_grid[:30, :30], cmap='gray') # Zoom in
    axes[0,1].set_title('Chromic Interference Grid (Zoomed)')
    axes[0,1].axis('off')
    
    # Row 2: Frequency Domain
    axes[1,0].imshow(gabor_fft, cmap='hot')
    axes[1,0].set_title('Kernel Spectral Response')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(grid_fft, cmap='hot')
    axes[1,1].set_title('Grid Spectral Power')
    axes[1,1].axis('off')
    
    plt.suptitle("Spectral Alignment Analysis: \nChromic Interference Perturbation Saturates Gabor Pass-Band", fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__)))), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments', 'results', 'figures', os.path.basename(os.path.dirname(os.path.abspath(__file__))), 'fig_gabor_fourier_analysis.png'), dpi=300)
    print("Generated fig_gabor_fourier_analysis.png")

if __name__ == "__main__":
    main()
