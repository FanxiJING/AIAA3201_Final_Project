#!/bin/bash
# =============================================================================
# part1_setup_and_run.sh
# Project 4 Part 1 — Full pipeline: install → data prep → COLMAP → 3DGS train
#
# Usage:
#   bash part1_setup_and_run.sh [--data-root <path>] [--output-root <path>]
#                               [--skip-install] [--skip-colmap] [--gpu <id>]
#
# Defaults:
#   --data-root    ./data
#   --output-root  ./output
#   --gpu          0
#
# Expected data layout (from your screenshot):
#   data/
#     405841/FRONT/rgb/          ← Waymo images
#     405841/FRONT/calib/        ← Waymo intrinsics
#     DL3DV-2/rgb/               ← DL3DV images
#     DL3DV-2/cameras.json
#     DL3DV-2/intrinsics.json
#     Re10k-1/images/            ← Re10k images
#     Re10k-1/cameras.json
#     Re10k-1/intrinsics.json
# =============================================================================

set -e
SECONDS=0

# ─── Color output helpers ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${BLUE}========== $* ==========${NC}\n"; }

# ─── Argument parsing ────────────────────────────────────────────────────────
DATA_ROOT="./data"
OUTPUT_ROOT="./output"
GPU_ID=0
SKIP_INSTALL=0
SKIP_COLMAP=0
USE_GPU=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)    DATA_ROOT="$2";    shift 2 ;;
        --output-root)  OUTPUT_ROOT="$2";  shift 2 ;;
        --gpu)          GPU_ID="$2";       shift 2 ;;
        --skip-install) SKIP_INSTALL=1;    shift ;;
        --skip-colmap)  SKIP_COLMAP=1;     shift ;;
        --cpu-only)     USE_GPU=0;         shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU_ID

# ─── Paths ───────────────────────────────────────────────────────────────────
WAYMO_DIR="$DATA_ROOT/405841/FRONT"
DL3DV_DIR="$DATA_ROOT/DL3DV-2"
RE10K_DIR="$DATA_ROOT/Re10k-1"
GS_REPO="./gaussian-splatting"
UTILS_DIR="./utils"
mkdir -p "$OUTPUT_ROOT" "$UTILS_DIR"

# =============================================================================
# SECTION 1 — INSTALL DEPENDENCIES
# =============================================================================
if [ "$SKIP_INSTALL" -eq 0 ]; then
    section "INSTALLING SYSTEM DEPENDENCIES"

    # Basic system packages
    info "Installing system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        git cmake ninja-build build-essential \
        libboost-all-dev libsuitesparse-dev libfreeimage-dev \
        libgoogle-glog-dev libgflags-dev libglew-dev \
        qtbase5-dev libqt5opengl5-dev libcgal-dev libcgal-qt5-dev \
        libatlas-base-dev libsuitesparse-dev \
        ffmpeg python3-pip python3-dev wget curl unzip \
        2>/dev/null || warn "Some apt packages may have failed — continuing"

    # ── COLMAP ───────────────────────────────────────────────────────────────
    section "INSTALLING COLMAP"
    if command -v colmap &>/dev/null; then
        success "COLMAP already installed: $(colmap --version 2>&1 | head -1)"
    else
        info "Building COLMAP from source..."
        if [ ! -d "colmap_src" ]; then
            git clone https://github.com/colmap/colmap.git colmap_src
        fi
        cd colmap_src
        mkdir -p build && cd build
        if [ "$USE_GPU" -eq 1 ]; then
            cmake .. -GNinja \
                -DCMAKE_CUDA_ARCHITECTURES=native \
                -DCMAKE_BUILD_TYPE=Release
        else
            cmake .. -GNinja \
                -DCUDA_ENABLED=OFF \
                -DCMAKE_BUILD_TYPE=Release
        fi
        ninja -j$(nproc)
        sudo ninja install
        cd ../..
        success "COLMAP installed."
    fi

    # ── Conda / Python env ───────────────────────────────────────────────────
    section "SETTING UP PYTHON ENVIRONMENT"
    if ! command -v conda &>/dev/null; then
        info "Installing Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda init bash
    else
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi

    if conda env list | grep -q "^gs "; then
        info "Conda env 'gs' already exists."
    else
        info "Creating conda env 'gs' (Python 3.8)..."
        conda create -n gs python=3.8 -y
    fi
    conda activate gs

    pip install -q numpy scipy pillow tqdm imageio \
        open3d pyquaternion

    # ── 3D Gaussian Splatting ─────────────────────────────────────────────────
    section "INSTALLING 3D GAUSSIAN SPLATTING"
    if [ ! -d "$GS_REPO" ]; then
        info "Cloning 3DGS repo..."
        git clone https://github.com/graphdeco-inria/gaussian-splatting \
            --recursive "$GS_REPO"
    else
        info "3DGS repo already exists."
    fi

    cd "$GS_REPO"
    pip install -q -r requirements.txt

    info "Installing submodule dependencies (diff-gaussian-rasterization, etc.)..."
    pip install -q submodules/diff-gaussian-rasterization
    pip install -q submodules/simple-knn
    cd ..
    success "3DGS installed."

else
    info "Skipping install (--skip-install set)."
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate gs 2>/dev/null || true
fi

# =============================================================================
# SECTION 2 — WRITE PYTHON HELPER SCRIPTS
# =============================================================================
section "WRITING HELPER SCRIPTS"

# ── json_to_colmap.py ─────────────────────────────────────────────────────────
cat > "$UTILS_DIR/json_to_colmap.py" << 'PYEOF'
"""
json_to_colmap.py
Converts cameras.json + intrinsics.json (DL3DV / Re10k format)
into COLMAP binary format (cameras.bin, images.bin, points3D.bin).

Usage:
    python utils/json_to_colmap.py \
        --dataset_dir data/DL3DV-2 \
        --image_subdir rgb \
        --output_dir data/DL3DV-2/sparse/0
"""

import argparse, json, os, struct, sys
import numpy as np

# ── COLMAP binary writers ────────────────────────────────────────────────────

def rotmat_to_quat(R):
    """Rotation matrix → quaternion (w, x, y, z)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def write_cameras_bin(path, width, height, fx, fy, cx, cy):
    """Write a single PINHOLE camera to cameras.bin."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))           # num_cameras = 1
        f.write(struct.pack("<I", 1))           # camera_id = 1
        f.write(struct.pack("<i", 1))           # model = PINHOLE
        f.write(struct.pack("<QQ", width, height))
        for p in [fx, fy, cx, cy]:
            f.write(struct.pack("<d", p))


def write_images_bin(path, frames):
    """
    frames: list of dicts with keys:
        name  (str)  — filename
        qvec  (4,)   — quaternion w,x,y,z  (world-to-camera)
        tvec  (3,)   — translation          (world-to-camera)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for i, fr in enumerate(frames):
            f.write(struct.pack("<I", i + 1))                   # image_id
            f.write(struct.pack("<dddd", *fr["qvec"]))           # quaternion
            f.write(struct.pack("<ddd",  *fr["tvec"]))           # translation
            f.write(struct.pack("<I", 1))                        # camera_id
            f.write(fr["name"].encode() + b"\x00")               # filename
            f.write(struct.pack("<Q", 0))                        # num_points2D = 0


def write_points3d_bin(path):
    """Write an empty points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ── Intrinsics loaders ───────────────────────────────────────────────────────

def load_intrinsics(path):
    """
    Handles several common formats:
      {"fx":..,"fy":..,"cx":..,"cy":..,"width":..,"height":..}
      {"fl_x":..,"fl_y":..,"cx":..,"cy":..,"w":..,"h":..}   (NeRF Studio)
      {"camera_intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]]}
    """
    with open(path) as f:
        d = json.load(f)

    # Try flat-key format first
    for fx_key in ["fx", "fl_x", "focal_x"]:
        if fx_key in d:
            fx = float(d[fx_key])
            fy = float(d.get("fy", d.get("fl_y", d.get("focal_y", fx))))
            cx = float(d["cx"])
            cy = float(d["cy"])
            w  = int(d.get("width", d.get("w", 0)))
            h  = int(d.get("height", d.get("h", 0)))
            return fx, fy, cx, cy, w, h

    # Try matrix format
    if "camera_intrinsics" in d:
        K = np.array(d["camera_intrinsics"])
        return K[0,0], K[1,1], K[0,2], K[1,2], int(d.get("width",0)), int(d.get("height",0))

    raise ValueError(f"Unrecognised intrinsics format in {path}.\n"
                     f"Keys found: {list(d.keys())}")


def load_cameras_json(path):
    """
    Handles several common camera pose formats.
    Returns list of {name, c2w (4x4 np.array)}.
    """
    with open(path) as f:
        data = json.load(f)

    # Format A: list of frames  [{"file_path":..., "transform_matrix":...}, ...]
    if isinstance(data, list):
        frames = data
        def parse(fr):
            name = os.path.basename(fr.get("file_path", fr.get("filename", fr.get("image", ""))))
            c2w  = np.array(fr.get("transform_matrix", fr.get("c2w", fr.get("extrinsic"))))
            return {"name": name, "c2w": c2w}
        return [parse(fr) for fr in frames]

    # Format B: {"frames": [...]}
    if "frames" in data:
        return load_cameras_json.__wrapped__({"frames": data["frames"]}) \
               if hasattr(load_cameras_json, "__wrapped__") else \
               [{"name": os.path.basename(fr.get("file_path","")),
                 "c2w": np.array(fr["transform_matrix"])}
                for fr in data["frames"]]

    # Format C: dict keyed by filename  {"img_001.jpg": [[...4x4...]], ...}
    if all(isinstance(v, list) for v in data.values()):
        return [{"name": k, "c2w": np.array(v)} for k, v in data.items()]

    raise ValueError(f"Unrecognised cameras.json format.\nKeys: {list(data.keys())[:8]}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",  required=True)
    ap.add_argument("--image_subdir", default="images",
                    help="Subfolder containing images, e.g. 'rgb' or 'images'")
    ap.add_argument("--output_dir",   default=None,
                    help="Defaults to <dataset_dir>/sparse/0")
    args = ap.parse_args()

    out = args.output_dir or os.path.join(args.dataset_dir, "sparse", "0")
    os.makedirs(out, exist_ok=True)

    intr_path = os.path.join(args.dataset_dir, "intrinsics.json")
    cam_path  = os.path.join(args.dataset_dir, "cameras.json")

    # ── Load intrinsics ──────────────────────────────────────────────────────
    print(f"Loading intrinsics from {intr_path}")
    fx, fy, cx, cy, w, h = load_intrinsics(intr_path)

    # If width/height not in intrinsics, read from first image
    if w == 0 or h == 0:
        img_dir = os.path.join(args.dataset_dir, args.image_subdir)
        first_img = sorted(os.listdir(img_dir))[0]
        from PIL import Image
        with Image.open(os.path.join(img_dir, first_img)) as im:
            w, h = im.size
    print(f"  Camera: {w}x{h}  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # ── Load camera poses ────────────────────────────────────────────────────
    print(f"Loading camera poses from {cam_path}")
    raw_frames = load_cameras_json(cam_path)
    print(f"  Found {len(raw_frames)} frames")

    # Convert c2w → w2c → quaternion + translation
    frames = []
    for fr in raw_frames:
        c2w = fr["c2w"]
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])  # make 4x4

        w2c = np.linalg.inv(c2w)
        R   = w2c[:3, :3]
        t   = w2c[:3,  3]
        q   = rotmat_to_quat(R)

        name = fr["name"]
        if not name:
            print(f"  WARNING: empty filename in entry {fr}, skipping")
            continue

        frames.append({"name": name, "qvec": q, "tvec": t})

    # ── Write COLMAP binaries ────────────────────────────────────────────────
    write_cameras_bin(f"{out}/cameras.bin", w, h, fx, fy, cx, cy)
    write_images_bin( f"{out}/images.bin",  frames)
    write_points3d_bin(f"{out}/points3D.bin")

    print(f"\nWrote COLMAP binaries to {out}/")
    print(f"  cameras.bin  — 1 PINHOLE camera")
    print(f"  images.bin   — {len(frames)} poses")
    print(f"  points3D.bin — empty (will be initialised by 3DGS)")
    print(f"\nNext step:")
    print(f"  python gaussian-splatting/train.py \\")
    print(f"      --source_path {args.dataset_dir} \\")
    print(f"      --images {args.image_subdir} \\")
    print(f"      --model_path ./output/<scene_name>")

if __name__ == "__main__":
    main()
PYEOF
success "Wrote utils/json_to_colmap.py"

# ── subsample.py ──────────────────────────────────────────────────────────────
cat > "$UTILS_DIR/subsample.py" << 'PYEOF'
"""
subsample.py — Create a sparse subset of images for Part 2.

Usage:
    python utils/subsample.py \
        --src  data/405841/FRONT/rgb \
        --dst  data/405841/FRONT/rgb_sparse \
        --step 10        # keep every 10th frame  (Waymo spec: 1/10)
"""
import argparse, os, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src",  required=True)
ap.add_argument("--dst",  required=True)
ap.add_argument("--step", type=int, default=10)
args = ap.parse_args()

exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
files = sorted([f for f in os.listdir(args.src)
                if Path(f).suffix in exts])
selected = files[::args.step]

os.makedirs(args.dst, exist_ok=True)
for f in selected:
    shutil.copy(os.path.join(args.src, f), os.path.join(args.dst, f))

print(f"Subsampled {len(files)} → {len(selected)} frames (step={args.step})")
print(f"Output: {args.dst}")
PYEOF
success "Wrote utils/subsample.py"

# =============================================================================
# SECTION 3 — DATA PREPARATION
# =============================================================================
section "PREPARING DATASETS"

# ── Helper: run COLMAP on a directory ────────────────────────────────────────
run_colmap() {
    local IMAGE_DIR=$1
    local OUT_DIR=$2
    local TAG=$3

    if [ "$SKIP_COLMAP" -eq 1 ] && [ -f "$OUT_DIR/sparse/0/cameras.bin" ]; then
        info "[$TAG] COLMAP output already exists, skipping."
        return
    fi

    local DATABASE="$OUT_DIR/database.db"
    mkdir -p "$OUT_DIR/sparse" "$OUT_DIR/dense"

    local NUM_IMAGES
    NUM_IMAGES=$(ls "$IMAGE_DIR" | wc -l)
    local MATCHER="exhaustive_matcher"
    if [ "$NUM_IMAGES" -gt 300 ]; then
        MATCHER="vocab_tree_matcher"
        if [ -z "$VOCAB_TREE_PATH" ]; then
            warn "[$TAG] >300 images but VOCAB_TREE_PATH not set. Downloading vocab tree..."
            wget -q https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin \
                -O "$DATA_ROOT/vocab_tree.bin"
            export VOCAB_TREE_PATH="$DATA_ROOT/vocab_tree.bin"
        fi
    fi

    info "[$TAG] $NUM_IMAGES images → using $MATCHER"

    # Stage 1: feature extraction
    info "[$TAG] Stage 1/4: feature_extractor"
    colmap feature_extractor \
        --database_path "$DATABASE" \
        --image_path "$IMAGE_DIR" \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu $USE_GPU \
        --SiftExtraction.max_num_features 8192

    # Stage 2: matching
    info "[$TAG] Stage 2/4: $MATCHER"
    if [ "$MATCHER" = "vocab_tree_matcher" ]; then
        colmap vocab_tree_matcher \
            --database_path "$DATABASE" \
            --SiftMatching.use_gpu $USE_GPU \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE_PATH"
    else
        colmap exhaustive_matcher \
            --database_path "$DATABASE" \
            --SiftMatching.use_gpu $USE_GPU
    fi

    # Stage 3: sparse reconstruction (SfM)
    info "[$TAG] Stage 3/4: mapper"
    colmap mapper \
        --database_path "$DATABASE" \
        --image_path "$IMAGE_DIR" \
        --output_path "$OUT_DIR/sparse" \
        --Mapper.num_threads "$(nproc)" \
        --Mapper.init_min_tri_angle 4 \
        --Mapper.multiple_models 0

    if [ ! -d "$OUT_DIR/sparse/0" ]; then
        error "[$TAG] COLMAP mapper produced no output. Check images and paths."
    fi

    NUM_MODELS=$(ls "$OUT_DIR/sparse" | wc -l)
    [ "$NUM_MODELS" -gt 1 ] && \
        warn "[$TAG] COLMAP produced $NUM_MODELS disconnected models — only sparse/0 used."

    # Stage 4: undistortion → 3DGS-ready format
    info "[$TAG] Stage 4/4: image_undistorter"
    colmap image_undistorter \
        --image_path "$IMAGE_DIR" \
        --input_path "$OUT_DIR/sparse/0" \
        --output_path "$OUT_DIR/dense" \
        --output_type COLMAP

    success "[$TAG] COLMAP done → $OUT_DIR/dense"
}

# ────────────────────────────────────────────────────────────────────────────
# WAYMO (405841)
# Has its own calib/ folder so COLMAP is the right approach.
# ────────────────────────────────────────────────────────────────────────────
section "WAYMO 405841 — data prep"

WAYMO_RGB="$WAYMO_DIR/rgb"
WAYMO_COLMAP_OUT="$WAYMO_DIR/colmap_out"
WAYMO_SPARSE_RGB="$WAYMO_DIR/rgb_sparse"

if [ ! -d "$WAYMO_RGB" ]; then
    error "Waymo rgb folder not found at $WAYMO_RGB"
fi

# Dense (Part 1): all frames
info "[Waymo] Preparing dense split..."
run_colmap "$WAYMO_RGB" "$WAYMO_COLMAP_OUT" "Waymo-dense"

# Sparse (Part 2): every 10th frame (BRPO spec)
info "[Waymo] Creating sparse split (1/10 frames)..."
python "$UTILS_DIR/subsample.py" \
    --src "$WAYMO_RGB" \
    --dst "$WAYMO_SPARSE_RGB" \
    --step 10

WAYMO_SPARSE_OUT="$WAYMO_DIR/colmap_sparse_out"
run_colmap "$WAYMO_SPARSE_RGB" "$WAYMO_SPARSE_OUT" "Waymo-sparse"

success "Waymo prepared."

# ────────────────────────────────────────────────────────────────────────────
# DL3DV-2
# Has cameras.json + intrinsics.json → skip COLMAP, convert directly.
# ────────────────────────────────────────────────────────────────────────────
section "DL3DV-2 — data prep"

if [ ! -d "$DL3DV_DIR" ]; then
    error "DL3DV-2 folder not found at $DL3DV_DIR"
fi

info "[DL3DV] Converting JSON poses → COLMAP binaries..."
python "$UTILS_DIR/json_to_colmap.py" \
    --dataset_dir "$DL3DV_DIR" \
    --image_subdir "rgb" \
    --output_dir "$DL3DV_DIR/sparse/0"

success "DL3DV-2 prepared."

# ────────────────────────────────────────────────────────────────────────────
# Re10k-1
# Same as DL3DV — JSON poses, no COLMAP needed.
# ────────────────────────────────────────────────────────────────────────────
section "Re10k-1 — data prep"

if [ ! -d "$RE10K_DIR" ]; then
    error "Re10k-1 folder not found at $RE10K_DIR"
fi

info "[Re10k] Converting JSON poses → COLMAP binaries..."
python "$UTILS_DIR/json_to_colmap.py" \
    --dataset_dir "$RE10K_DIR" \
    --image_subdir "images" \
    --output_dir "$RE10K_DIR/sparse/0"

success "Re10k-1 prepared."

# =============================================================================
# SECTION 4 — 3DGS TRAINING
# =============================================================================
section "3DGS TRAINING — PLAN A (COLMAP init)"

# Common 3DGS flags
GS_TRAIN="python $GS_REPO/train.py"
BASE_FLAGS="--iterations 30000 --eval"

# ── Waymo dense ──────────────────────────────────────────────────────────────
info "[3DGS] Waymo dense / Plan A..."
$GS_TRAIN \
    --source_path "$WAYMO_COLMAP_OUT/dense" \
    --model_path  "$OUTPUT_ROOT/waymo_planA" \
    $BASE_FLAGS \
    | tee "$OUTPUT_ROOT/waymo_planA_train.log"
success "[3DGS] Waymo Plan A done."

# ── DL3DV dense ──────────────────────────────────────────────────────────────
info "[3DGS] DL3DV-2 dense / Plan A..."
$GS_TRAIN \
    --source_path "$DL3DV_DIR" \
    --model_path  "$OUTPUT_ROOT/dl3dv_planA" \
    --images      "rgb" \
    $BASE_FLAGS \
    | tee "$OUTPUT_ROOT/dl3dv_planA_train.log"
success "[3DGS] DL3DV Plan A done."

# ── Re10k dense ──────────────────────────────────────────────────────────────
info "[3DGS] Re10k-1 dense / Plan A..."
$GS_TRAIN \
    --source_path "$RE10K_DIR" \
    --model_path  "$OUTPUT_ROOT/re10k_planA" \
    --images      "images" \
    $BASE_FLAGS \
    | tee "$OUTPUT_ROOT/re10k_planA_train.log"
success "[3DGS] Re10k Plan A done."

# =============================================================================
# SECTION 5 — EVALUATION
# =============================================================================
section "EVALUATING RESULTS"

# Write a quick eval script inline if it doesn't exist yet
EVAL_SCRIPT="$UTILS_DIR/evaluate.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
cat > "$EVAL_SCRIPT" << 'PYEOF'
"""
evaluate.py — Compute PSNR / SSIM / LPIPS on 3DGS test renders.
Usage:
    python utils/evaluate.py --model_path output/waymo_planA
"""
import argparse, json, os
from pathlib import Path
import torch
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import torchvision.transforms.functional as TF

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True,
                help="Path to trained 3DGS model output dir")
ap.add_argument("--render_dir", default=None,
                help="Defaults to <model_path>/train/ours_30000/renders")
args = ap.parse_args()

render_dir = args.render_dir or os.path.join(
    args.model_path, "train", "ours_30000", "renders")
gt_dir = render_dir.replace("/renders", "/gt")

if not os.path.isdir(render_dir):
    print(f"Render dir not found: {render_dir}")
    print("Run gaussian-splatting/render.py first.")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

renders = sorted(Path(render_dir).glob("*.png"))
psnr_vals, ssim_vals, lpips_vals = [], [], []

for r in renders:
    gt_path = Path(gt_dir) / r.name
    if not gt_path.exists():
        continue
    pred = TF.to_tensor(Image.open(r).convert("RGB")).unsqueeze(0).to(device)
    gt   = TF.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
    psnr_vals.append(psnr_m(pred, gt).item())
    ssim_vals.append(ssim_m(pred, gt).item())
    lpips_vals.append(lpips_m(pred, gt).item())

results = {
    "num_images": len(psnr_vals),
    "PSNR":  sum(psnr_vals)  / len(psnr_vals),
    "SSIM":  sum(ssim_vals)  / len(ssim_vals),
    "LPIPS": sum(lpips_vals) / len(lpips_vals),
}
print(json.dumps(results, indent=2))
out_path = os.path.join(args.model_path, "metrics.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
PYEOF
fi

for MODEL in waymo_planA dl3dv_planA re10k_planA; do
    MODEL_PATH="$OUTPUT_ROOT/$MODEL"
    if [ -d "$MODEL_PATH" ]; then
        info "Rendering $MODEL for evaluation..."
        python "$GS_REPO/render.py" \
            --model_path "$MODEL_PATH" \
            --skip_train 2>/dev/null || warn "render.py failed for $MODEL, skipping eval"

        python "$EVAL_SCRIPT" \
            --model_path "$MODEL_PATH" || warn "eval failed for $MODEL"
    fi
done

# =============================================================================
# DONE
# =============================================================================
section "ALL DONE"
ELAPSED=$SECONDS
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))
success "Total time: ${MINS}m ${SECS}s"

echo ""
echo "Output summary:"
echo "  Waymo  Plan A model  → $OUTPUT_ROOT/waymo_planA/"
echo "  DL3DV  Plan A model  → $OUTPUT_ROOT/dl3dv_planA/"
echo "  Re10k  Plan A model  → $OUTPUT_ROOT/re10k_planA/"
echo ""
echo "  Metrics (after eval) → <model_path>/metrics.json"
echo ""
echo "Next: run Plan B (foundation model init) and compare convergence curves."
echo "      See part1_dense/run_foundation.py"