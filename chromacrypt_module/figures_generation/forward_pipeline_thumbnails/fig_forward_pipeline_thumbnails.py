import os
"""
Generates thumbnail images for Figure 3: Oklch Forward Transformation Pipeline.

PURPOSE: Demonstrate PERCEPTUAL UNIFORMITY — equal angular steps in Oklch hue
produce equal perceived colour changes. Equal steps in sRGB do NOT.

Pipeline (5 steps, 6 representations):
  sRGB → Linear RGB → LMS → LMS' → Oklab → Oklch

The reference is a standard sRGB colour wheel (hue angles sample the sRGB cube
edges — perceptually non-uniform: green dominates, blue is compressed).

The final Oklch thumbnail is a DIFFERENT colour wheel generated directly in
Oklch hue space (L=0.72 fixed, C varies radially) and then converted BACK to
sRGB via the inverse Oklch transform — this makes the perceptual uniformity
visually obvious: hues are evenly spread with equal perceptual weight.
"""

import numpy as np
import cv2

SIZE = 256

# ─── Forward transform matrices (Björn Ottosson 2020) ────────────────────────
M1 = np.array([
    [0.4121656120, 0.5362752080, 0.0514575653],
    [0.2118591070, 0.6807189584, 0.1074065790],
    [0.0883097947, 0.2818474174, 0.6302613616],
])
M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
])

M1_inv = np.linalg.inv(M1)
M2_inv = np.linalg.inv(M2)


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def normalize(img, lo=2, hi=98):
    """Per-channel percentile stretch to vivid uint8 (for intermediate stages)."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[..., c].astype(np.float32)
        p_lo, p_hi = np.percentile(ch, lo), np.percentile(ch, hi)
        out[..., c] = np.clip((ch - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


# ─── Stage 0: sRGB colour wheel (hue in sRGB space) ─────────────────────────
def make_srgb_wheel(size=256):
    """Standard HSV colour wheel — hue angles sample sRGB cube edges.
    This is perceptually NON-uniform: green dominates ~1/3 of the circle."""
    cx, cy = size // 2, size // 2
    Y, X   = np.mgrid[0:size, 0:size]
    dx, dy = (X - cx).astype(float), (cy - Y).astype(float)
    radius = np.sqrt(dx**2 + dy**2)
    angle  = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    max_r  = size // 2 - 2
    mask   = radius <= max_r

    H = (angle / 2).astype(np.float32)
    S = (np.clip(radius / max_r, 0, 1) * 255).astype(np.float32)
    V = np.full_like(H, 255, dtype=np.float32)

    bgr = cv2.cvtColor(np.stack([H, S, V], axis=-1).astype(np.uint8),
                       cv2.COLOR_HSV2BGR)
    out      = np.full((size, size, 3), 200, dtype=np.uint8)
    out[mask] = bgr[mask]
    return out


# ─── Stage 5: Oklch colour wheel (hue in Oklch space, back-projected to sRGB) ─
def make_oklch_wheel(size=256, L=0.72, C_max=0.15):
    """Colour wheel sampled uniformly in Oklch hue angle and converted to sRGB.
    Because Oklch hue is PERCEPTUALLY UNIFORM, all colours appear equally spaced
    and equally bright — no colour dominates the circle.
    """
    cx, cy = size // 2, size // 2
    Y, X   = np.mgrid[0:size, 0:size]
    dx, dy = (X - cx).astype(float), (cy - Y).astype(float)
    radius = np.sqrt(dx**2 + dy**2)
    H      = np.arctan2(dy, dx)           # Oklch hue angle (note: dy up is positive)
    max_r  = size // 2 - 2
    mask   = radius <= max_r

    # Oklch coordinates
    C = C_max * np.clip(radius / max_r, 0, 1)

    # Oklch → Oklab
    a = C * np.cos(H)
    b = C * np.sin(H)
    # Oklab → LMS'
    oklab = np.stack([np.full_like(a, L), a, b], axis=-1)
    lms_prime = oklab @ M2_inv.T
    # LMS' → LMS
    lms = lms_prime ** 3
    # LMS → Linear RGB
    linear = lms @ M1_inv.T
    # Linear RGB → sRGB
    srgb = np.clip(linear_to_srgb(linear), 0, 1)
    srgb_u8 = (srgb * 255).astype(np.uint8)

    out      = np.full((size, size, 3), 200, dtype=np.uint8)
    out[mask] = cv2.cvtColor(srgb_u8, cv2.COLOR_RGB2BGR)[mask]
    return out


# ────────────────────────────────────────────────────────────────────────────
# Generate all thumbnails
# ────────────────────────────────────────────────────────────────────────────

# Stage 0 — sRGB input
wheel_srgb = make_srgb_wheel(SIZE)
cv2.imwrite('fig_thumb_rgb.png', wheel_srgb)

# Float pipeline input
rgb = cv2.cvtColor(wheel_srgb, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

# Stage 1 — Linearise (inverse gamma) — display with gamma re-encoded
linear = srgb_to_linear(rgb)
cv2.imwrite('fig_thumb_linear_rgb.png',
            cv2.cvtColor(
                (np.clip(linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR))

# Stage 2 — LMS  (adaptive normalize for vivid display)
lms = linear @ M1.T
cv2.imwrite('fig_thumb_lms.png', cv2.cvtColor(normalize(lms), cv2.COLOR_RGB2BGR))

# Stage 3 — LMS'  (adaptive normalize)
lms_prime = np.cbrt(lms)
cv2.imwrite('fig_thumb_lms_prime.png', cv2.cvtColor(normalize(lms_prime), cv2.COLOR_RGB2BGR))

# Stage 4 — Oklab  (L→R, a→G, b→B, adaptive normalize)
oklab = lms_prime @ M2.T
oklab_raw = np.stack([oklab[..., 0], oklab[..., 1] + 0.5, oklab[..., 2] + 0.5], axis=-1)
cv2.imwrite('fig_thumb_oklab.png', cv2.cvtColor(normalize(oklab_raw), cv2.COLOR_RGB2BGR))

# Stage 5 — Oklch colour wheel rendered in OKLCH SPACE → back to sRGB
# This directly demonstrates perceptual uniformity vs the sRGB wheel above
wheel_oklch = make_oklch_wheel(SIZE, L=0.72, C_max=0.15)
cv2.imwrite('fig_thumb_oklch.png', wheel_oklch)


print("✓ Thumbnails generated — perceptual uniformity demo:")
print("   thumb_rgb.png         — sRGB hue wheel (NON-uniform: green dominates)")
print("   thumb_linear_rgb.png  — Linearised sRGB (gamma removed)")
print("   thumb_lms.png         — LMS cone space (M1 projection)")
print("   thumb_lms_prime.png   — LMS' cube-root compression")
print("   thumb_oklab.png       — Oklab projection (M2, Lab channels)")
print("   thumb_oklch.png       — Oklch hue wheel (UNIFORM: all hues equally spaced)")
