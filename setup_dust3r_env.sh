#!/usr/bin/env bash
# setup_dust3r_env.sh
# Run with the "dust3r" conda env active:
#   conda activate dust3r
#   bash setup_dust3r_env.sh
#
# What this does:
#   1. Builds 3DGS CUDA submodules (diff-gaussian-rasterization, simple-knn)
#   2. Installs DUSt3R as an editable Python package (and builds curope)
#   3. Installs the few missing pip packages (open3d)
#   4. Clones RegGS
#   5. Downloads DUSt3R weights
#
# What this does NOT touch:
#   - colmap        (already installed: 3.10 GPU, conda-forge)
#   - torch 2.11.0  (already installed)
#   - numpy / pandas / matplotlib / einops / roma / trimesh / tqdm / etc.
#   - dust3r repo   (already cloned)
#   - gaussian-splatting repo (already cloned)

set -e

# ── Paths — edit if your layout differs ───────────────────────────────────────
WORKPLACE="/data2/fjing221/workplace"
DATA_ROOT="/data2/fjing221/data"
WEIGHTS="${WORKPLACE}/weights"

GS_REPO="${WORKPLACE}/gaussian-splatting"
DUST3R_REPO="${WORKPLACE}/dust3r"
REGGS_REPO="${WORKPLACE}/RegGS"

DUST3R_CKPT="${WEIGHTS}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

OUTPUT_PART1="${WORKPLACE}/output/part1"
OUTPUT_PART2="${WORKPLACE}/output/part2"

echo "=================================================="
echo "  Project 4 — Environment Setup (dust3r env)"
echo "  Torch : $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA  : $(python -c 'import torch; print(torch.version.cuda)')"
echo "  nvcc  : $(nvcc --version 2>/dev/null | grep release || echo 'not in PATH')"
echo "=================================================="

# ── 1. Create output dirs ──────────────────────────────────────────────────────
mkdir -p "${WEIGHTS}" \
         "${OUTPUT_PART1}/figures" \
         "${OUTPUT_PART2}/figures"

# ── 2. Install DUSt3R as editable package ─────────────────────────────────────
echo ""
# Instead of: pip install -e ${DUST3R_REPO}
# Do this:

echo "[1/4] Setting up DUSt3R (path-based, no pip install)..."

# Install DUSt3R's actual dependencies from its requirements.txt
if [ -f "${DUST3R_REPO}/requirements.txt" ]; then
    pip install -r "${DUST3R_REPO}/requirements.txt" -q
    echo "  requirements.txt installed."
fi

# Install croco sub-package if it has a setup file
if [ -f "${DUST3R_REPO}/croco/setup.py" ] || [ -f "${DUST3R_REPO}/croco/pyproject.toml" ]; then
    pip install -e "${DUST3R_REPO}/croco" --no-build-isolation -q 2>/dev/null || true
fi

# Build curope C extension (optional speedup)
CUROPE="${DUST3R_REPO}/croco/models/curope"
if [ -d "${CUROPE}" ]; then
    echo "  Building curope extension..."
    (cd "${CUROPE}" && python setup.py build_ext --inplace 2>/dev/null) \
        && echo "  curope built OK." \
        || echo "  curope build skipped (OK — pure Python fallback used)."
fi

# Add to PYTHONPATH permanently for this shell session
export PYTHONPATH="${DUST3R_REPO}:${DUST3R_REPO}/croco:${PYTHONPATH}"
echo "  PYTHONPATH updated."

# Verify
python -c "from dust3r.model import AsymmetricCroCo3DStereo; print('  dust3r import OK')"

# ── 3. Build 3DGS CUDA submodules ─────────────────────────────────────────────
echo ""
echo "[2/4] Building 3DGS CUDA submodules..."
echo "  (uses torch 2.11.0 + conda cuda-nvcc 12.6 — ~3-5 min)"

if python -c "from diff_gaussian_rasterization import GaussianRasterizer" 2>/dev/null; then
    echo "  diff-gaussian-rasterization already installed."
else
    echo "  Building diff-gaussian-rasterization..."
    pip install "${GS_REPO}/submodules/diff-gaussian-rasterization" \
        --no-build-isolation -q
    echo "  diff-gaussian-rasterization done."
fi

if python -c "import simple_knn" 2>/dev/null; then
    echo "  simple-knn already installed."
else
    echo "  Building simple-knn..."
    pip install "${GS_REPO}/submodules/simple-knn" \
        --no-build-isolation -q
    echo "  simple-knn done."
fi

# ── 4. Install remaining pip packages ─────────────────────────────────────────
echo ""
echo "[3/4] Installing remaining pip packages..."
# open3d is the main missing one; the rest are already present
pip install open3d -q
echo "  Done."

# ── 5. Clone RegGS ────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Cloning RegGS..."
if [ -d "${REGGS_REPO}" ]; then
    echo "  RegGS already present: ${REGGS_REPO}"
else
    git clone https://github.com/chengchong01/RegGS "${REGGS_REPO}"
    echo "  RegGS cloned."
fi

# ── 5. Download DUSt3R weights ────────────────────────────────────────────────
echo ""
echo "[5/5] DUSt3R weights..."
if [ -f "${DUST3R_CKPT}" ]; then
    SIZE=$(du -sh "${DUST3R_CKPT}" | cut -f1)
    echo "  Already present (${SIZE}): ${DUST3R_CKPT}"
else
    echo "  Downloading DUSt3R ViT-Large (~2.5 GB)..."
    wget --show-progress \
         "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
         -O "${DUST3R_CKPT}"
    echo "  Saved → ${DUST3R_CKPT}"
fi

# ── Verify ─────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Verification"
echo "=================================================="
python - <<'PYEOF'
import sys
results = []

def chk(label, fn):
    try:
        fn()
        results.append(f"  OK      {label}")
    except Exception as e:
        results.append(f"  MISSING {label}  ({e})")

chk("torch + CUDA",         lambda: __import__('torch').cuda.is_available() or (_ for _ in ()).throw(RuntimeError("no CUDA")))
chk("dust3r",               lambda: __import__('dust3r'))
chk("diff_gaussian_rast.",  lambda: __import__('diff_gaussian_rasterization'))
chk("simple_knn",           lambda: __import__('simple_knn'))
chk("colmap (subprocess)",  lambda: __import__('subprocess').run("colmap --version", shell=True, check=True, capture_output=True))
chk("open3d",               lambda: __import__('open3d'))
chk("einops",               lambda: __import__('einops'))
chk("roma",                 lambda: __import__('roma'))
chk("trimesh",              lambda: __import__('trimesh'))
chk("tensorboard",          lambda: __import__('tensorboard'))

for r in results:
    print(r)
PYEOF

echo ""
echo "=================================================="
echo "  Setup complete!"
echo ""
echo "  Run Part 1:  python part1_linux.py"
echo "  Run Part 2:  python part2_linux.py"
echo "=================================================="
