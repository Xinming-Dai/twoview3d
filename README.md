# twoview3d

A two-view 3D reconstruction pipeline that includes [E-RayZer](E-RayZer/) for self-supervised 3D reconstruction from two images.

## Repository Structure
Set up the repository structure as follows

```
project3d/
├── twoview3d/
│   └── src/
│       └── twoview3d/
│           ├── data/       # Bundle adjustment, calibration
│           ├── preprocess/ # Frame extraction, resize-to-match utilities
│           └── model/      # Point cloud and 3D model utilities
├── E-RayZer/              # 3D reconstruction (Gaussian splatting)
└── other_model/           # other 3D models that can be used for comparison
```

## Getting Started

### 1. Fork repositories
Fork the `twoview3d` and `E-RayZer` repositories to your own GitHub account.

### 2. Clone the repository (with submodules)

```bash
cd project3d
cd twoview3d
git clone <your-twoview3d-repo-url>
git clone --recurse-submodules <your-E-RayZer-repo-url>
cd E-RayZer
```

### 2. Environment setup

This project uses a single environment for both twoview3d and E-RayZer (Python 3.10 required):

```bash
conda create -n project3d python=3.10 -y
conda activate project3d

# Install dependencies
pip install -e E-RayZer/
pip install -r E-RayZer/requirements.txt # GPU requirements
# pip install -r E-RayZer/requirements-cpu.txt # if developing on CPU
pip install -e twoview3d/
pip install -r twoview3d/requirements.txt

# Install gsplat (E-RayZer's Gaussian splatting backend; takes several minutes)
pip install -e E-RayZer/third_party/gsplat/
# BUILD_NO_CUDA=1 pip install -e E-RayZer/third_party/gsplat/ # if developing on CPU
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
