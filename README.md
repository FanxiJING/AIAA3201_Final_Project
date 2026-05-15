# Project 4: Generative Sparse-View 3D Reconstruction
### AIAA 3201 — Introduction to Computer Vision, Spring 2026

> **Parts 1 & 2 implementation** — Initialization analysis (COLMAP vs. DUSt3R)
> and unposed sparse reconstruction (DUSt3R-only · InstantSplat · RegGS-style).

---

## Table of Contents
1. [Overview](#overview)
2. [Results Summary](#results-summary)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Data Preparation](#data-preparation)
6. [Part 1: High-Fidelity Reconstruction](#part-1-high-fidelity-reconstruction)
7. [Part 2: Unposed Sparse Reconstruction](#part-2-unposed-sparse-reconstruction)
8. [Part 3: Generative Enhancement](#part-3-generative-enhancement)
9. [Output Structure](#output-structure)
10. [Key Findings](#key-findings)
11. [Troubleshooting](#troubleshooting)
12. [Citation](#citation)

---

## Overview

This project reconstructs 3D scenes using 3D Gaussian Splatting (3DGS) under three conditions:

| Part | Condition | Poses | Frames | Method |
|------|-----------|-------|--------|--------|
| **Part 1** | Dense | Known (COLMAP or DUSt3R) | ~50 | Plan A: COLMAP -> 3DGS · Plan B: DUSt3R -> 3DGS |
| **Part 2** | Sparse | Unknown | 9-19 | DUSt3R-only · InstantSplat-style · RegGS-style |
| **Part 3** | Sparse + Generative | Unknown | 9-19 + pseudo | InstantSplat + DIFIX3D+ pseudo-views |

**Datasets:** Waymo-405841 (outdoor), DL3DV-2 (outdoor), Re10k-1 (indoor/outdoor).

---

## Results Summary

### Part 1 — COLMAP vs. DUSt3R Initialization (30k iterations, training split)

| Dataset | Plan A PSNR↑ | Plan A SSIM↑ | Plan B PSNR↑ | Plan B SSIM↑ | ΔPSNR (B−A) |
|---------|-------------|-------------|-------------|-------------|-------------|
| Waymo   | **39.99**   | **0.963**   | 24.87       | 0.796       | −15.12      |
| DL3DV   | **32.77**   | **0.970**   | 29.02       | 0.916       | −3.76       |
| Re10k   | **36.36**   | **0.989**   | 33.37       | 0.983       | −2.99       |
| **Avg** | **36.38**   | **0.974**   | 29.09       | 0.898       | −7.29       |

COLMAP initialization outperforms DUSt3R by **3.0–15.1 dB** across all datasets.

### Part 2 — Unposed Sparse Reconstruction (30k iterations, test split)

| Dataset | Method | PSNR↑ | SSIM↑ | ATE↓ (m) |
|---------|--------|-------|-------|----------|
| Waymo (1/10, 19 fr.) | DUSt3R-only | **14.56** | **0.489** | 0.291 |
| | InstantSplat | 14.59 | 0.481 | 0.292 |
| | RegGS-style | 13.75 | 0.463 | 0.295 |
| DL3DV (1/30, 10 fr.) | DUSt3R-only | **11.32** | 0.222 | **0.809** |
| | InstantSplat | 11.30 | **0.225** | **0.809** |
| | RegGS-style | 8.09 | 0.141 | 0.810 |
| Re10k (1/30, 9 fr.) | DUSt3R-only | 9.39 | 0.218 | 1.914 |
| | InstantSplat | **9.41** | **0.220** | 1.914 |
| | RegGS-style | 8.99 | 0.181 | **1.908** |

> **Key finding:** RegGS-style alignment is *worse* than the plain baseline (−0.4 to −3.2 dB) due to over-regularisation at extreme sparsity. InstantSplat changes quality by ≤0.04 dB with identical ATE, confirming pose estimation — not point-cloud initialisation — is the bottleneck.

---

## Repository Structure

```
project4/
├── README.md
│
├── setup_dust3r_env.sh        # One-time environment + dependency setup
│
├── part1_linux.py             # Part 1: COLMAP vs. DUSt3R -> 3DGS
├── part2_original_data.py     # Part 2: DUSt3R-only sparse reconstruction
├── part2_alternative.py       # Part 2: InstantSplat & RegGS-style variants
│
├── part3_step1_render_intermediate.py   # Part 3: Pose interpolation + rendering
├── part3_step2_difix3d.py              # Part 3: DIFIX3D+ enhancement
├── part3_step3_rerun.py                # Part 3: Hybrid training with pseudo-views
│
└── (generated at runtime)
    ├── data/                  # COLMAP-processed dense datasets
    │   ├── 405841/dense/0/
    │   ├── DL3DV-2/dense/0/
    │   └── Rek10-v1/dense/0/
    └── workplace/
        ├── gaussian-splatting/ # 3DGS repo (cloned by setup)
        ├── dust3r/             # DUSt3R repo (pre-cloned)
        ├── RegGS/              # RegGS repo (cloned by setup)
        ├── weights/
        │   └── DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
        └── output/
            ├── part1/          # Part 1 results
            ├── part2_original/ # Part 2 DUSt3R-only results
            ├── part2_alternative/ # Part 2 InstantSplat + RegGS results
            └── part3_enhanced/    # Part 3 generative enhancement results
```

---

## Environment Setup

### Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with ≥16 GB VRAM (tested on A100 / V100)
- CUDA 12.x + conda

### Step 1 — Create conda environment

```bash
conda create -n dust3r python=3.10 -y
conda activate dust3r

# Install PyTorch (adjust cuda version to match your driver)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install COLMAP via conda-forge (GPU build)
conda install -c conda-forge colmap -y

# Core scientific stack
pip install numpy pillow matplotlib pandas tensorboard einops roma trimesh tqdm
```

### Step 2 — Clone required repositories

```bash
cd /data2/fjing221/workplace   # or your preferred workplace directory

git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
git clone https://github.com/naver/dust3r
```

### Step 3 — Run the automated setup script

This script builds CUDA submodules, installs DUSt3R dependencies, clones RegGS,
and downloads the DUSt3R weights (~2.5 GB):

```bash
conda activate dust3r
bash setup_dust3r_env.sh
```

What the script does:
- Installs DUSt3R requirements from its `requirements.txt`
- Builds the `curope` C extension (optional speedup, falls back to pure Python)
- Builds 3DGS CUDA submodules (`diff-gaussian-rasterization`, `simple-knn`)
- Clones RegGS into `workplace/RegGS/`
- Downloads `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` to `workplace/weights/`

### Step 4 — Verify installation

```bash
python - <<'EOF'
import torch
from dust3r.model import AsymmetricCroCo3DStereo
from diff_gaussian_rasterization import GaussianRasterizer
import simple_knn
print("All imports OK")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

---

## Data Preparation

### Expected raw data layout

```
/data2/fjing221/
    project4/                      # raw sequences (Part 2 input)
        405841/FRONT/rgb/          # Waymo: 199 frames
        DL3DV-2/rgb/               # DL3DV: 306 frames
        Re10k-1/images/            # Re10k: 280 frames
    data/                          # COLMAP-processed (Part 1 input)
        405841/dense/0/
            images/                # undistorted images
            sparse/0/              # cameras.bin, images.bin, points3D.bin
        DL3DV-2/dense/0/
            images/
            sparse/0/
        Rek10-v1/dense/0/
            images/
            sparse/0/
```

### Download links

| Dataset | Source | Notes |
|---------|--------|-------|
| Waymo-405841 | [Baidu](https://pan.baidu.com) / [Google Drive](https://drive.google.com) | Outdoor driving sequence |
| DL3DV-2 | [Baidu](https://pan.baidu.com) / [Google Drive](https://drive.google.com) | Outdoor scene |
| Re10k-1 | [Baidu](https://pan.baidu.com) / [Google Drive](https://drive.google.com) | Indoor scene |

> Download links provided on the course Canvas page.

### Verify data paths

```bash
# Part 1 check
python part1_linux.py --datasets waymo dl3dv re10k --skip_planA --skip_planB

# Part 2 check — prints image counts
python part2_original_data.py --datasets waymo dl3dv re10k 2>&1 | head -20
```

---

## Part 1: High-Fidelity Reconstruction

**Script:** `part1_linux.py`

Compares two initialization strategies for 3DGS on ~50-frame dense sequences:
- **Plan A:** COLMAP SfM → sparse point cloud → 3DGS
- **Plan B:** DUSt3R zero-shot pose estimation → dense point cloud → 3DGS

### Run — both plans, all datasets

```bash
conda activate dust3r
python part1_linux.py
```

### Run — custom options

```bash
python part1_linux.py \
    --datasets waymo dl3dv re10k \   # subset: e.g. --datasets waymo
    --iterations 30000 \              # 3DGS training iterations
    --dust3r_img_size 512 \           # DUSt3R image resolution
    --dust3r_niter 300 \              # DUSt3R global alignment iterations
    --skip_planA \                    # skip COLMAP plan (run DUSt3R only)
    --skip_planB                      # skip DUSt3R plan (run COLMAP only)
```

### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `/data2/fjing221/data` | Root of COLMAP-processed datasets |
| `--workplace` | `/data2/fjing221/workplace` | Repos, weights, outputs |
| `--datasets` | `waymo dl3dv re10k` | Which datasets to process |
| `--iterations` | `30000` | 3DGS training iterations |
| `--skip_planA` | `False` | Skip COLMAP pipeline |
| `--skip_planB` | `False` | Skip DUSt3R pipeline |
| `--dust3r_img_size` | `512` | Image size for DUSt3R inference |
| `--dust3r_niter` | `300` | DUSt3R global alignment iterations |

### Pipeline steps (Plan A)

```
images/ + sparse/0/  →  3DGS train.py (30k iters)  →  render.py  →  PSNR/SSIM
```

### Pipeline steps (Plan B)

```
images/  →  DUSt3R inference  →  global_aligner (Sim3)
         →  COLMAP binary format  →  3DGS train.py  →  render.py  →  PSNR/SSIM
```

### Outputs

```
workplace/output/part1/
    {dataset}/
        planA/3dgs/                # COLMAP-initialized 3DGS model
        planB/3dgs/                # DUSt3R-initialized 3DGS model
    metrics.json                   # PSNR/SSIM for all plans and datasets
    part1_table.csv                # Summary comparison table
    figures/
        convergence.png            # PSNR vs. iteration for Plan A vs B
```

### Expected runtime

| Stage | Time |
|-------|------|
| COLMAP (per dataset, ~50 images) | 5–15 min |
| DUSt3R inference + alignment | 3–8 min |
| 3DGS training (30k iters, GPU) | 20–40 min per model |
| Rendering + evaluation | 2–5 min |

---

## Part 2: Unposed Sparse Reconstruction

Two scripts cover Part 2. Run `part2_original_data.py` first (DUSt3R-only
baseline), then `part2_alternative.py` for the InstantSplat and RegGS variants.

---

### Part 2a — DUSt3R-only Baseline

**Script:** `part2_original_data.py`

Sub-samples to 1/10–1/30 sparsity, runs DUSt3R global alignment with no pose
input, trains 3DGS, and evaluates PSNR/SSIM on the test split plus ATE vs.
COLMAP ground truth.

#### Run

```bash
conda activate dust3r
python part2_original_data.py
```

#### Custom options

```bash
python part2_original_data.py \
    --datasets waymo dl3dv re10k \
    --iterations 30000 \
    --dust3r_img_size 512 \
    --dust3r_niter 300
```

#### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--project4` | `/data2/fjing221/project4` | Raw full-sequence image root |
| `--workplace` | `/data2/fjing221/workplace` | Repos, weights, outputs |
| `--datasets` | `waymo dl3dv re10k` | Datasets to process |
| `--iterations` | `30000` | 3DGS training iterations |
| `--dust3r_img_size` | `512` | DUSt3R image resolution |
| `--dust3r_niter` | `300` | Global alignment iterations |

#### Sparsity ratios applied

| Dataset | Total frames | Step | Sparse frames |
|---------|-------------|------|---------------|
| Waymo-405841 | 199 | 10 | 19 |
| DL3DV-2 | 306 | 30 | 10 |
| Re10k-1 | 280 | 30 | 9 |

#### Pipeline

```
full sequence  →  subsample (1/step)  →  DUSt3R inference (complete/swin graph)
              →  global_aligner CPU (Sim3, 300 iters)
              →  COLMAP binary format  →  3DGS (30k iters)
              →  render (test split)  →  PSNR/SSIM  →  ATE vs COLMAP GT
```

#### Outputs

```
workplace/output/part2_original/
    {dataset}/
        sparse/0/           # DUSt3R-recovered COLMAP format
        images/             # symlink to sparse image dir
        poses_c2w.npy       # recovered camera poses (N,4,4)
        3dgs_model/         # trained 3DGS model
        metrics.json        # PSNR, SSIM, ATE_RMSE
    figures/
        {dataset}_trajectory.png    # 3D camera trajectory plot
    part2_original_results.csv      # full comparison table
```

---

### Part 2b — InstantSplat & RegGS-style Variants

**Script:** `part2_alternative.py`

Runs the same sparse subsampling but with two enhanced reconstruction strategies,
then assembles a comparison table alongside the Part 2a baseline.

**This script is fully resume-safe:** every stage (reconstruction, training,
rendering, evaluation) checks for existing outputs and skips if already complete.
Previously finished results are never overwritten.

#### Run — both methods

```bash
conda activate dust3r
python part2_alternative.py --method both
```

#### Run — single method

```bash
# InstantSplat only
python part2_alternative.py --method instantsplat

# RegGS-style only
python part2_alternative.py --method reggs
```

#### Run — single dataset

```bash
python part2_alternative.py --method both --datasets re10k
```

#### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--project4` | `/data2/fjing221/project4` | Raw full-sequence image root |
| `--workplace` | `/data2/fjing221/workplace` | Repos, weights, outputs |
| `--data_root` | `/data2/fjing221/data` | COLMAP GT root for ATE |
| `--datasets` | `waymo dl3dv re10k` | Datasets to process |
| `--method` | `both` | `reggs` / `instantsplat` / `both` |
| `--iterations` | `30000` | 3DGS training iterations |
| `--dust3r_img_size` | `512` | DUSt3R image resolution |
| `--dust3r_niter` | `300` | DUSt3R global alignment iterations |
| `--reg_niter` | `500` | RegGS total alignment iters (¼ warm-up + full refine) |

#### Method descriptions

**InstantSplat variant**
Runs standard DUSt3R global alignment (identical poses to baseline), then applies
confidence-weighted point sampling per view: up to 8,000 points are sampled
weighted by proximity to the scene median depth, preferring geometrically stable
regions. This produces a higher-quality seeding point cloud for 3DGS while
keeping pose estimates unchanged.

**RegGS-style two-phase alignment**
Phase 1 (linear LR schedule, lr = 0.05, `niter/4` iters): MST initialisation for
rapid coarse alignment. Phase 2 (cosine LR, lr = 0.01, `niter` iters): refinement
on the same scene object without re-initialisation, followed by scale
normalisation (mean inter-camera distance → 1).

> **Implementation note:** calling `compute_global_alignment(init="known_poses")`
> in Phase 2 raises `"not all poses are known"` because this mode requires
> externally pre-fixed pose tensors. The fix is `init="mst"` on both phases;
> the optimizer detects existing parameters on the second call and continues
> from the current state.

#### Outputs

```
workplace/output/part2_alternative/
    {dataset}/
        instantsplat/
            sparse/0/               # COLMAP format from InstantSplat
            images/                 # symlink
            poses_c2w.npy           # recovered poses (N,4,4)
            3dgs/                   # trained 3DGS model
            instantsplat_metrics.json
        reggs/
            sparse/0/
            images/
            poses_c2w.npy
            3dgs/
            reggs_metrics.json
    comparison_table.csv            # all methods × all datasets
    all_results.json
    figures/
        {dataset}_trajectory_compare.png   # side-by-side trajectory plots
```
---

## Part 3: Generative Enhancement

**Scripts:** `part3_step1_render_intermediate.py`, `part3_step2_difix3d.py`, `part3_step3_rerun.py`

Explores whether diffusion-generated pseudo-views can improve unposed sparse reconstruction by augmenting the training set with intermediate views. This experiment builds on the optimal configuration from Part 2 (InstantSplat) and applies DIFIX3D+ — a single-step diffusion model fine-tuned for removing 3D reconstruction artifacts.

### 3.1 Methodology

The pipeline consists of three sequential steps, each implemented in a separate script.

#### Step 1 — Intermediate Pose Rendering (`part3_step1_render_intermediate.py`)

Given sparse camera poses recovered by DUSt3R in Part 2 (9–11 frames per scene), this script generates intermediate camera poses between consecutive real frames using spherical linear interpolation (SLERP) for rotation and linear interpolation for translation. For each interval between two real frames, we generate K = 2 intermediate poses, yielding approximately 18–20 pseudo-views per scene.

For each interpolated pose, the script renders a novel view using the 3DGS model trained in Part 2 (InstantSplat). These renders typically contain floating artifacts, blurring, and incomplete geometry due to insufficient multi-view constraints. The rendered images are saved to `intermediate_renders/` directory.

#### Step 2 — DIFIX3D+ Enhancement (`part3_step2_difix3d.py`)

This script applies DIFIX3D+ [Wu et al., CVPR 2025], a single-step diffusion model built on SD-Turbo, to enhance the rendered views. For each rendered image, the model takes the degraded render and the nearest real frame as reference, and outputs an enhanced pseudo-view in a single denoising step. The enhanced images are saved to `pseudo_views/` directory with confidence scores stored as separate `.npy` files.

#### Step 3 — Hybrid Training (`part3_step3_rerun.py`)

This script concatenates the generated pseudo-views with the original real images to form an augmented training set, then re-runs the complete Part 2 pipeline (DUSt3R + 3DGS) on this augmented dataset. The output is saved to a separate directory (`part3_alternative/`) to avoid overwriting Part 2 results.


---

## Output Structure (complete)

```
workplace/
└── output/
    ├── part1/
    │   ├── waymo/
    │   │   ├── planA/3dgs/        # COLMAP → 3DGS
    │   │   │   ├── point_cloud/iteration_30000/
    │   │   │   └── train/ours_30000/{renders,gt}/
    │   │   └── planB/
    │   │       ├── dust3r_init/   # DUSt3R COLMAP-format output
    │   │       └── 3dgs/
    │   ├── dl3dv/ ...
    │   ├── re10k/ ...
    │   ├── metrics.json
    │   ├── part1_table.csv
    │   └── figures/convergence.png
    │
    ├── part2_original/
    │   ├── waymo/
    │   │   ├── sparse/0/
    │   │   ├── poses_c2w.npy
    │   │   ├── 3dgs_model/
    │   │   │   └── test/ours_30000/{renders,gt}/
    │   │   └── metrics.json
    │   ├── dl3dv/ ...
    │   ├── re10k/ ...
    │   ├── part2_original_results.csv
    │   └── figures/
    │       ├── waymo_trajectory.png
    │       ├── dl3dv_trajectory.png
    │       └── re10k_trajectory.png
    │
    └── part2_alternative/
    |   │   ├── waymo/
    │   │   ├── instantsplat/
    │   │   └── reggs/
    │   ├── dl3dv/ ...
    │   ├── re10k/ ...
    │   ├── comparison_table.csv
    │   ├── all_results.json
    │   └── figures/
    │       ├── waymo_trajectory_compare.png
    │       ├── dl3dv_trajectory_compare.png
    │       └── re10k_trajectory_compare.png
    │
    └── part3_enhanced/
        ├── dl3dv/
        │   ├── instantsplat/
        │   │   ├── intermediate_renders/
        │   │   │   ├── 00000.png
        │   │   │   ├── 00001.png
        │   │   │   └── ...
        │   │   ├── pseudo_views/
        │   │   │   ├── pseudo_00000.png
        │   │   │   ├── pseudo_00000_confidence.npy
        │   │   │   ├── pseudo_00001.png
        │   │   │   ├── pseudo_00001_confidence.npy
        │   │   │   └── ...
        │   │   ├── intermediate_poses.npy
        │   │   ├── 3dgs_retrained/
        │   │   │   └── test/ours_30000/{renders,gt}/
        │   │   └── metrics.json
        │   └── reggs/ ...
        ├── re10k/ ...
        ├── waymo/ ...
        ├── part3_results.csv
        └── figures/
            ├── dl3dv_pseudo_comparison.png
            ├── re10k_pseudo_comparison.png
            └── waymo_pseudo_comparison.png
```

---

## Key Findings

### Part 1
- COLMAP initialization outperforms DUSt3R by **3.0–15.1 dB** PSNR on all
  datasets and metrics.
- The gap scales with **scene scale and baseline width**: Waymo (outdoor,
  wide baseline) suffers −15.1 dB; Re10k (indoor, bounded) only −3.0 dB.
- DUSt3R still converges to 24.9–33.4 dB, making it a viable fallback for
  bounded scenes where COLMAP fails (textureless surfaces, very few views).

### Part 2
- All methods degrade **21–27 dB** below the dense COLMAP reference at 1/10–1/30
  sparsity with no pose input.
- **RegGS-style alignment is worse than the plain baseline** on all three
  datasets (−0.4 to −3.2 dB PSNR, −0.026 to −0.081 SSIM), with similar or
  higher ATE. Two-phase regularisation over-constrains the geometry when only
  9–10 frames are available.
- **InstantSplat-style sampling is neutral** (±0.04 dB, ≤0.001 m ATE change),
  confirming that the bottleneck is **pose estimation accuracy**, not
  point-cloud initialization quality.
- Meaningful progress requires better pose estimators (monocular SLAM,
  feed-forward pose-free models) rather than better initialization pipelines.

### Part 3
We evaluate on DL3DV and Re10k using the InstantSplat configuration (the best-performing method from Part 2). The sparsity ratios are 1/30 for both datasets, resulting in 10–11 real frames per scene. Table 5 presents the quantitative comparison.

**Table 5:** Generative enhancement results (test split, 30k iterations).

| Dataset | Method | PSNR↑ | SSIM↑ | ATE (m)↓ |
|--------|--------|-------|-------|----------|
| DL3DV | Part 2 (InstantSplat) | 11.30 | 0.225 | 0.809 |
| | Part 3 (w/ pseudo-views) | 9.20 | 0.136 | 0.833 |
| Re10k | Part 2 (InstantSplat) | 9.41 | 0.220 | 1.914 |
| | Part 3 (w/ pseudo-views) | 7.98 | 0.105 | 1.918 |

Adding pseudo-views leads to consistent degradation: PSNR drops by 2.10 dB (18.6%) on DL3DV and 1.43 dB (15.2%) on Re10k. SSIM decreases significantly, while ATE remains nearly unchanged, confirming that pose estimation is not the source of the degradation.

---

## Troubleshooting

### Not all poses are known
**Cause:** `compute_global_alignment(init="known_poses")` was called in Phase 2
of the RegGS-style alignment. This mode requires externally pre-fixed pose
tensors and does not accept MST-estimated poses.  
**Fix:** Already corrected in `part2_alternative.py` — both phases use
`init="mst"`. The optimizer continues from the Phase 1 parameter state on the
second call without re-initialising.

### COLMAP `feature_extractor` flag errors
**Cause:** Flag names changed between COLMAP versions.  
**Fix:** `part1_linux.py` auto-detects correct flags by parsing `--help` output
at startup. Check the printed `COLMAP 3.10 flags` block to verify detection.

### GPU OOM during DUSt3R global alignment
**Cause:** `PointCloudOptimizer` holds all pair outputs in memory.  
**Fix:** Both Part 2 scripts move all tensors to CPU before alignment
(`_move_to_cpu(output)`). If OOM still occurs during inference, reduce
`--dust3r_img_size` to `224` or `336`.

### 3DGS training crash (`point_cloud/iteration_N` missing)
**Cause:** `train.py` failed silently.  
**Fix:** Re-run with `--iterations 30000`; all scripts detect incomplete
training (missing `point_cloud/iteration_N/`) and re-run automatically.
Check that `diff-gaussian-rasterization` and `simple-knn` built correctly
(run `python -c "from diff_gaussian_rasterization import GaussianRasterizer"`).

### `curope` build failure
**Cause:** CUDA toolkit version mismatch.  
**Fix:** The setup script catches this and falls back to the pure-Python
implementation automatically. Performance is slightly lower but correctness
is unaffected.

### Zero PSNR / all-black renders
**Cause:** Empty `points3D.bin` or corrupt COLMAP sparse directory.  
**Fix:** `part1_linux.py` auto-creates an empty `points3D.bin` if missing.
If the issue persists, delete `sparse/0/` and re-run to regenerate.

### ATE is `None` / GT not found
**Cause:** COLMAP GT `images.bin` path does not exist (dense reconstruction not
yet run, or path mismatch).  
**Fix:** Verify paths in `DATASETS_ALL["gt_poses"]` match your actual COLMAP
output directory. ATE evaluation is skipped gracefully if GT is absent.

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{kerbl20233dgs,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas
          and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42}, number={4}, pages={139:1--139:14}, year={2023}
}

@inproceedings{wang2024dust3r,
  title={{DUSt3R}: Geometric {3D} Vision Made Easy},
  author={Wang, Shuzhe and Leroy, Vincent and Cabon, Yohann and
          Chidlovskii, Boris and Revaud, Jerome},
  booktitle={CVPR}, pages={20697--20709}, year={2024}
}

@inproceedings{cheng2025reggs,
  title={{RegGS}: Unposed Sparse Views Gaussian Splatting with {3DGS} Registration},
  author={Cheng, Chong and Hu, Yu and Yu, Sicheng and Zhao, Beizhen and
          Wang, Zijian and Wang, Hao},
  booktitle={ICCV}, pages={8100--8109}, year={2025}
}

@article{fan2024instantsplat,
  title={{InstantSplat}: Sparse-View Gaussian Splatting in Seconds},
  author={Fan, Zhiwen and Cong, Wenyan and Wen, Kairun and others},
  journal={arXiv:2403.20309}, year={2024}
}
```

---

*AIAA 3201, Spring 2026. Report available in `project4_report_final.pdf`.*