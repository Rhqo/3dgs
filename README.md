# Rendering 3DGS

Origin : https://github.com/KAIST-Visual-AI-Group/CS479-Assignment-3DGS \
Download dataset : https://drive.google.com/file/d/14YVFRR-8L8UVR_UXOe_W-ogNs0IM0572/view

## Quick Start
```bash
uv run render.py --scene-type lego --device-type cuda --out-root ./outputs

# uv run render.py -h
╭─ options ────────────────────────────────────────────────────────────────────╮
│ -h, --help             show this help message and exit                       │
│ --scene-type {chair,drums,ficus,hotdog,lego,materials,mic,ship}              │
│                        Type of scene to render. (default: lego)              │
│ --device-type {cpu,cuda}                                                     │
│                        Device to use for rendering. (default: cuda)          │
│ --out-root PATH        Root directory for saving outputs. (default: outputs) │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Code Structure
```bash
gs_renderer
│
├── data                <- Directory for data files.
├── src
│   ├── camera.py       <- A light-weight data class for storing camera parameters.
│   ├── renderer.py     <- Main renderer implementation.
│   ├── scene.py        <- A light-weight data class for storing Gaussian Splat parameters.
│   └── sh.py           <- A utility for processing Spherical Harmonic coefficients.
├── evaluate.py         <- Script for computing evaluation metrics.
├── render.py           <- Main script for rendering.
├── render_all.sh       <- Shell script for rendering all scenes for evaluation.
└── README.md           <- This file.
```

## Data Structure
```
data
│
├── nerf_synthetic      <- Directory containing camera parameters and reference images
│   ├── chair
│   ├── drums
│   ├── lego
│   └── materials
├── chair.ply           <- Gaussian splats for "Chair" Scene.
├── drums.ply           <- Gaussian splats for "Drums" Scene.
├── lego.ply            <- Gaussian splats for "Lego" Scene.
└── materials.ply       <- Gaussian splats for "Materials" Scene.
```
