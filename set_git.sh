#!/usr/bin/env bash
# =============================================================================
# fix_and_push.sh  —  Run this from /data2/fjing221 to finish the GitHub push.
# The previous script's commit was skipped because globs weren't expanded.
# This script stages files correctly and pushes via SSH.
#
# Usage:
#   cd /data2/fjing221
#   bash fix_and_push.sh
# =============================================================================

set -e
cd /data2/fjing221
echo "Working in: $(pwd)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP A — Expand .gitignore with extra patterns for files showing up untracked
# ─────────────────────────────────────────────────────────────────────────────
echo "[A] Updating .gitignore with additional patterns..."

cat >> .gitignore << 'EOF'

# ── TensorBoard event files (large, not needed for reproduction) ──────────────
events.out.tfevents.*

# ── 3DGS config args files ────────────────────────────────────────────────────
cfg_args

# ── Symlinked image directories inside output (point to raw data) ─────────────
workplace/output/**/images

# ── COLMAP sparse binary dirs inside output ───────────────────────────────────
workplace/output/**/sparse/

# ── NumPy pose files (large, reproducible by re-running) ─────────────────────
workplace/output/**/*.npy

# ── Video files ───────────────────────────────────────────────────────────────
*.mp4
*.avi
*.mov

# ── DUSt3R init dirs (contain large binary COLMAP files) ─────────────────────
workplace/output/**/dust3r_init/

# ── Scratch files ─────────────────────────────────────────────────────────────
hello_world.txt
report.html
EOF

echo "    .gitignore updated OK"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP B — Stage exactly the files we want using find (no glob expansion issues)
# ─────────────────────────────────────────────────────────────────────────────
echo "[B] Staging files with find (glob-safe)..."

# 1. Core scripts and config
for f in \
    .gitignore \
    .gitmodules \
    README.md \
    setup_dust3r_env.sh \
    set_git.sh \
    part1_linux.py \
    part2_original_data.py \
    part2_alternative.py
do
    if [ -f "$f" ]; then
        git add "$f"
        echo "    staged: $f"
    else
        echo "    skip (not found): $f"
    fi
done

# 2. Output results — metrics JSON files
echo ""
echo "    Staging JSON metrics..."
find workplace/output -name "*.json" -not -path "*/point_cloud/*" \
    | while read f; do
        git add "$f"
        echo "    + $f"
    done

# 3. Output results — CSV tables
echo ""
echo "    Staging CSV tables..."
find workplace/output -name "*.csv" \
    | while read f; do
        git add "$f"
        echo "    + $f"
    done

# 4. Output results — PNG figures only
echo ""
echo "    Staging PNG figures..."
find workplace/output -name "*.png" \
    | while read f; do
        git add "$f"
        echo "    + $f"
    done

echo ""
echo "    All target files staged."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP C — Show what is staged vs still untracked
# ─────────────────────────────────────────────────────────────────────────────
echo "[C] Current git status:"
echo ""
echo "  === STAGED (will be committed) ==="
git diff --cached --name-only | sed 's/^/    /'
echo ""
echo "  === STILL UNTRACKED (will NOT be committed) ==="
git status --short | grep "^??" | sed 's/^/    /'
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP D — Commit
# ─────────────────────────────────────────────────────────────────────────────
STAGED_COUNT=$(git diff --cached --name-only | wc -l)

if [ "$STAGED_COUNT" -eq 0 ]; then
    echo "[D] Nothing staged to commit — all target files may already be committed."
    echo "    Proceeding directly to push..."
else
    echo "[D] Committing $STAGED_COUNT files..."
    git commit -m "Project 4: Part 1 & 2 implementation + results

Scripts:
- part1_linux.py: COLMAP (Plan A) vs DUSt3R (Plan B) 3DGS initialization
- part2_original_data.py: DUSt3R-only unposed sparse reconstruction
- part2_alternative.py: InstantSplat and RegGS-style variants
- setup_dust3r_env.sh: one-command environment setup

Part 1 PSNR results (30k iters, training split):
  Waymo  COLMAP 39.99 dB vs DUSt3R 24.87 dB  (delta -15.12)
  DL3DV  COLMAP 32.77 dB vs DUSt3R 29.02 dB  (delta -3.76)
  Re10k  COLMAP 36.36 dB vs DUSt3R 33.37 dB  (delta -2.99)

Part 2 PSNR results (30k iters, test split, unposed):
  Waymo  DUSt3R-only 14.56 | InstantSplat 14.59 | RegGS 13.75
  DL3DV  DUSt3R-only 11.32 | InstantSplat 11.30 | RegGS  8.09
  Re10k  DUSt3R-only  9.39 | InstantSplat  9.41 | RegGS  8.99

Submodules: dust3r, gaussian-splatting, RegGS, vggt"
    echo "    Commit OK"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP E — Push via SSH
# ─────────────────────────────────────────────────────────────────────────────
echo "[E] Pushing to git@github.com:FanxiJING/AIAA3201_Final_Project.git ..."
git push origin main
echo ""
echo "============================================================"
echo "  Push complete!"
echo "  View at: https://github.com/FanxiJING/AIAA3201_Final_Project"
echo "============================================================"