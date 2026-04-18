# ChromaCrypt

**Chromic Interference & Adversarial Geometry Bounding Framework**

This repository contains the standalone, formal implementation of mathematical structures mapped in the NeurIPS submission "Chromic Interference". It guarantees exact geometric bounds via 12-step recursive gamut limits across OKLCH and Fourier planes.

## Architecture

*   **`attacks.py`**: Isolates optical gradient math (`LuminancePGD`, `ChromicPGD`) and structural topologies (`NarrowbandMimicry`, `TopologicalAttractor`). Features native 12-step subset preservation arrays enforcing RGB boundary projection cleanly.
*   **`color_ops.py`**: Differentiable conversion mapping between bounds (`L`, `C`, `H` extraction) optimizing cleanly across `.float16` constraint surfaces.

## Executing Benchmarks (Reproducibility)

The central manuscript data is generated explicitly through these 4 execution frameworks:

1.  Structure a local environment seamlessly to capture required dependencies (`diffusers`, `lpips`, `torch`):
    ```bash
    setup_venv_win.bat
    ```
2.  Navigate to `/experiments/` and execute target mapping arrays natively:
    ```bash
    python benchmark_transferability.py
    python benchmark_robust_models.py
    python benchmark_structural_mechanisms.py
    python generate_manuscript_figures.py
    ```

> Execution requires local validation models and ImageNet batches loaded into `/data/imagenet-1k/` relative to project root.

*Note: Diffusion purification leverages the `madebyollin/sdxl-vae-fp16-fix` decoder internally to prevent latent tensor overflow natively.*
