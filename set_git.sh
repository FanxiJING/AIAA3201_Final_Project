#!/usr/bin/env bash
# =============================================================================
# final_push.sh  —  Complete, self-contained GitHub push script.
# Run from anywhere; it always operates in /data2/fjing221.
#
# Usage:
#   bash final_push.sh
# =============================================================================

set -e
PROJECT_ROOT="/data2/fjing221"
REMOTE="git@github.com:FanxiJING/AIAA3201_Final_Project.git"

cd "$PROJECT_ROOT"
echo "=================================================="
echo "  GitHub push — AIAA3201 Final Project"
echo "  Working in: $(pwd)"
echo "=================================================="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Write .gitignore from scratch (overwrites any previous version)
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/5] Writing .gitignore..."

cat > .gitignore << 'GITIGNORE'
# ── Conda / pip / torch caches ────────────────────────────────────────────────
conda_envs/
conda_pkgs/
pip_cache/
torch_cache/
huggingface_cache/

# ── Scratch / personal ────────────────────────────────────────────────────────
tmp/
lxy/
project4/
hello_world.txt
report.html

# ── Raw datasets (too large — see README for download links) ──────────────────
data/

# ── Third-party tool directories (not your code) ─────────────────────────────
workplace/colmap/
workplace/test_gs/

# ── Large model weight files ──────────────────────────────────────────────────
workplace/weights/*.pth
workplace/weights/*.pt
workplace/weights/*.bin
workplace/weights/*.safetensors

# ── 3DGS large binary outputs ────────────────────────────────────────────────
workplace/output/**/point_cloud/
workplace/output/**/*.ply
workplace/output/**/*.splat

# ── Render directories (large images — PNG figures are force-added below) ─────
workplace/output/**/renders/
workplace/output/**/gt/

# ── DUSt3R init dirs (contain large COLMAP binary files) ─────────────────────
workplace/output/**/dust3r_init/

# ── COLMAP sparse binary dirs inside output ───────────────────────────────────
workplace/output/**/sparse/

# ── Symlinked image directories inside output (point to raw data) ─────────────
workplace/output/**/images

# ── NumPy pose files (reproducible by re-running scripts) ────────────────────
workplace/output/**/*.npy

# ── TensorBoard event logs ────────────────────────────────────────────────────
events.out.tfevents.*

# ── 3DGS config args ──────────────────────────────────────────────────────────
cfg_args

# ── Video files ───────────────────────────────────────────────────────────────
*.mp4
*.avi
*.mov

# ── Python build artifacts ────────────────────────────────────────────────────
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# ── Editor / OS ───────────────────────────────────────────────────────────────
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db
GITIGNORE

echo "    Done."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Stage core project files
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/5] Staging core project files..."

# .gitignore and submodule config
git add .gitignore
git add .gitmodules 2>/dev/null && echo "    + .gitmodules" || true

# Every .py and .sh file in the project root
for f in "$PROJECT_ROOT"/*.py "$PROJECT_ROOT"/*.sh; do
    [ -f "$f" ] || continue
    git add "$f"
    echo "    + $(basename $f)"
done

# README
if [ -f README.md ]; then
    git add README.md
    echo "    + README.md"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Stage output results (metrics JSON, CSV tables, PNG figures)
#           Use -f (force) for PNGs so renders/ gitignore pattern doesn't block
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/5] Staging output results..."

# JSON metrics files
JSON_COUNT=0
while IFS= read -r -d '' f; do
    git add "$f"
    JSON_COUNT=$((JSON_COUNT + 1))
done < <(find workplace/output -name "*.json" -not -path "*/point_cloud/*" -print0)
echo "    + $JSON_COUNT JSON files"

# CSV tables
CSV_COUNT=0
while IFS= read -r -d '' f; do
    git add "$f"
    CSV_COUNT=$((CSV_COUNT + 1))
done < <(find workplace/output -name "*.csv" -print0)
echo "    + $CSV_COUNT CSV files"

# PNG figures — force-add so renders/ gitignore rule doesn't block them
PNG_COUNT=0
while IFS= read -r -d '' f; do
    git add -f "$f"
    PNG_COUNT=$((PNG_COUNT + 1))
done < <(find workplace/output -name "*.png" -print0)
echo "    + $PNG_COUNT PNG files (force-added)"

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Show summary and commit
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/5] Commit summary:"
STAGED=$(git diff --cached --name-only)
STAGED_COUNT=$(echo "$STAGED" | grep -c . || true)
echo ""
echo "$STAGED" | sed 's/^/    /'
echo ""
echo "    Total: $STAGED_COUNT files staged"
echo ""

if [ "$STAGED_COUNT" -eq 0 ]; then
    echo "    Nothing new to commit — already up to date."
else
    git commit -m "Project 4: Part 1 & 2 — scripts, results, figures

Scripts:
  part1_linux.py          — COLMAP vs DUSt3R 3DGS initialization (Part 1)
  part2_original_data.py  — DUSt3R-only unposed sparse reconstruction (Part 2)
  part2_alternative.py    — InstantSplat and RegGS-style variants (Part 2)
  setup_dust3r_env.sh     — one-command environment setup

Part 1 results (PSNR, 30k iters, train split):
  Waymo  COLMAP 39.99 dB  DUSt3R 24.87 dB  delta -15.12
  DL3DV  COLMAP 32.77 dB  DUSt3R 29.02 dB  delta  -3.76
  Re10k  COLMAP 36.36 dB  DUSt3R 33.37 dB  delta  -2.99

Part 2 results (PSNR, 30k iters, test split, unposed):
  Waymo  baseline 14.56  InstantSplat 14.59  RegGS 13.75
  DL3DV  baseline 11.32  InstantSplat 11.30  RegGS  8.09
  Re10k  baseline  9.39  InstantSplat  9.41  RegGS  8.99

Submodules: dust3r, gaussian-splatting, RegGS, vggt"
    echo "    Commit created."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Push
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/5] Pushing to GitHub..."
git remote set-url origin "$REMOTE" 2>/dev/null || git remote add origin "$REMOTE"
git push origin main
echo ""
echo "=================================================="
echo "  Done! View at:"
echo "  https://github.com/FanxiJING/AIAA3201_Final_Project"
echo "=================================================="