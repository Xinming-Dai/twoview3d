# twoview3d

A two-view 3D reconstruction pipeline that includes [E-RayZer](E-RayZer/) for self-supervised 3D reconstruction from two images.

## Repository Structure

```
twoview3d/
├── data/           # Bundle adjustment, calibration
├── preprocess/     # Frame extraction, resize-to-match utilities
├── model/          # Point cloud and 3D model utilities
└── E-RayZer/       # Submodule: 3D reconstruction (Gaussian splatting)
```

## Getting Started

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules <your-twoview3d-repo-url>
cd twoview3d
```

### 2. Environment setup

This project uses a single environment for both twoview3d and E-RayZer (Python 3.10 required):

```bash
conda create -n twoview3d python=3.10 -y
conda activate twoview3d

# Install dependencies
pip install -r requirements.txt
pip install -r E-RayZer/requirements.txt

# Install gsplat (E-RayZer's Gaussian splatting backend; takes several minutes)
pip install -e E-RayZer/third_party/gsplat/
```

### 3. Run E-RayZer demo

```bash
cd E-RayZer
python gradio_app.py \
  --config config/erayzer.yaml \
  --ckpt checkpoints/erayzer_multi.pt \
  --device cuda:0 \
  --output-dir outputs \
  --share
```

See [E-RayZer/README.md](E-RayZer/README.md) for more details.

## Git: Submodule Workflows

**Update submodule to latest on main:**

```bash
cd E-RayZer && git pull origin main && cd ..
```

**Pin submodule to a specific commit (recommended for reproducibility):**

```bash
cd E-RayZer
git checkout <commit-hash>
cd ..
git add E-RayZer
git commit -m "Pin E-RayZer to specific commit"
```
