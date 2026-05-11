#!/usr/bin/env python3
"""
Project 4 — Part 2: Unposed Sparse Reconstruction (Original Full Datasets)
Uses the original ~200-300 frame sequences with correct sparsity ratios:
    Waymo  405841     : 199 frames, step=10 → 19 sparse frames  (1/10)
    DL3DV  DL3DV-2   : 306 frames, step=30 → 10 sparse frames  (1/30)
    Re10k  Re10k-1   : 280 frames, step=30 →  9 sparse frames  (1/30)

Pipeline (Option A — DUSt3R):
    Sparse images (NO camera poses used)
        ↓ DUSt3R pairwise pointmap prediction
        ↓ global_aligner Sim3 registration (on CPU to avoid OOM)
        ↓ Unified point cloud + recovered camera poses
        ↓ 3DGS post-optimization
        ↓ Evaluate: PSNR / SSIM + ATE RMSE vs COLMAP GT

Usage:
    conda activate dust3r
    python part2_original_data.py [--datasets waymo dl3dv re10k]
"""

import os, sys, json, struct, time, subprocess, argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--project4",   default="/data2/fjing221/project4")
parser.add_argument("--workplace",  default="/data2/fjing221/workplace")
parser.add_argument("--datasets", nargs="+",
                    default=["waymo", "dl3dv", "re10k"],
                    choices=["waymo", "dl3dv", "re10k"])
parser.add_argument("--iterations",      type=int, default=30000)
parser.add_argument("--dust3r_img_size", type=int, default=512)
parser.add_argument("--dust3r_niter",    type=int, default=300)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT4    = args.project4
WORKPLACE   = args.workplace
WEIGHTS_DIR = f"{WORKPLACE}/weights"
OUTPUT_ROOT = f"{WORKPLACE}/output/part2_original"
# GT poses from the 50-frame COLMAP-processed subset (Part 1 data)
DATA_ROOT   = "/data2/fjing221/data"

GS_REPO     = f"{WORKPLACE}/gaussian-splatting"
DUST3R_REPO = f"{WORKPLACE}/dust3r"
REGGS_REPO  = f"{WORKPLACE}/RegGS"
DUST3R_CKPT = f"{WEIGHTS_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# ── Dataset configs ───────────────────────────────────────────────────────────
# Images from original full sequences — NO poses provided to reconstruction
# GT poses from the 50-frame COLMAP subset for ATE evaluation
DATASETS_ALL = {
    "waymo": {
        "images"   : f"{PROJECT4}/405841/FRONT/rgb",
        "step"     : 10,    # 199/10 = 19 sparse frames — follows 1/10 spec
        "gt_poses" : f"{DATA_ROOT}/405841/dense/0/sparse/0/images.bin",
        "total"    : 199,
    },
    "dl3dv": {
        "images"   : f"{PROJECT4}/DL3DV-2/rgb",
        "step"     : 30,    # 306/30 = 10 sparse frames — exact 1/30 spec
        "gt_poses" : f"{DATA_ROOT}/DL3DV-2/dense/0/sparse/0/images.bin",
        "total"    : 306,
    },
    "re10k": {
        "images"   : f"{PROJECT4}/Re10k-1/images",
        "step"     : 30,    # 280/30 = 9 sparse frames — follows 1/30 spec
        "gt_poses" : f"{DATA_ROOT}/Rek10-v1/dense/0/sparse/0/images.bin",
        "total"    : 280,
    },
}
DATASETS = {k: v for k, v in DATASETS_ALL.items() if k in args.datasets}

# ── Create output dirs ────────────────────────────────────────────────────────
for d in [OUTPUT_ROOT, f"{OUTPUT_ROOT}/figures"]:
    os.makedirs(d, exist_ok=True)

# ── Add repos to Python path ──────────────────────────────────────────────────
for p in [GS_REPO, DUST3R_REPO, f"{DUST3R_REPO}/croco", REGGS_REPO]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ── Environment check ─────────────────────────────────────────────────────────
print("=" * 60)
print("  Part 2 — Unposed Sparse Reconstruction (Original Data)")
print("=" * 60)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nPyTorch : {torch.__version__}  CUDA: {torch.version.cuda}")
print(f"Device  : {DEVICE}  GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'N/A'}")
print(f"\nProject4: {PROJECT4}")
print(f"Output  : {OUTPUT_ROOT}")

print("\nDataset image dirs:")
for name, cfg in DATASETS.items():
    exists = os.path.isdir(cfg["images"])
    n = len([f for f in os.listdir(cfg["images"])
             if not os.path.isdir(os.path.join(cfg["images"], f))]) if exists else 0
    sparse_n = n // cfg["step"]
    print(f"  {name:8s}: {n:4d} total → {sparse_n:3d} sparse "
          f"(1/{cfg['step']})  {'OK' if exists else 'MISSING'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(img_dir, step=1):
    files = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if not os.path.isdir(os.path.join(img_dir, f))
        and Path(f).suffix.lower() in IMG_EXTS
    ])
    return files[::step]

def rotmat_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s,
                         (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                         0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                         (R[1,2]+R[2,1])/s, 0.25*s])

def write_cameras_bin(path, w, h, fx, fy, cx, cy):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<QQ", int(w), int(h)))
        for p in [fx, fy, cx, cy]:
            f.write(struct.pack("<d", float(p)))

def write_images_bin(path, frames):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for i, fr in enumerate(frames):
            f.write(struct.pack("<I", i+1))
            f.write(struct.pack("<dddd", *fr["qvec"]))
            f.write(struct.pack("<ddd",  *fr["tvec"]))
            f.write(struct.pack("<I", 1))
            f.write(fr["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", 0))

def write_points3d_bin(path, pts=None):
    with open(path, "wb") as f:
        if pts is None:
            f.write(struct.pack("<Q", 0))
        else:
            f.write(struct.pack("<Q", len(pts)))
            for i, (xyz, rgb) in enumerate(pts):
                f.write(struct.pack("<Q", i+1))
                f.write(struct.pack("<ddd", *xyz))
                f.write(struct.pack("<BBB", *[int(c) for c in rgb]))
                f.write(struct.pack("<d", 0.0))
                f.write(struct.pack("<Q", 0))

import torch.nn.functional as F
import torchvision.transforms.functional as TF

def compute_psnr(p, g):
    return (10 * torch.log10(torch.tensor(1.0) / F.mse_loss(p, g))).item()

def compute_ssim(p, g, k=11):
    C1, C2 = 0.01**2, 0.03**2
    m1 = F.avg_pool2d(p, k, 1, k//2); m2 = F.avg_pool2d(g, k, 1, k//2)
    s1  = F.avg_pool2d(p*p, k, 1, k//2) - m1**2
    s2  = F.avg_pool2d(g*g, k, 1, k//2) - m2**2
    s12 = F.avg_pool2d(p*g, k, 1, k//2) - m1*m2
    return (((2*m1*m2+C1)*(2*s12+C2))/((m1**2+m2**2+C1)*(s1+s2+C2))).mean().item()

def evaluate_model(model_path):
    iters = args.iterations
    # Prefer test set; fall back to train if test is empty
    for split in ["test", "train"]:
        rd = f"{model_path}/{split}/ours_{iters}/renders"
        gd = f"{model_path}/{split}/ours_{iters}/gt"
        if not os.path.isdir(rd):
            continue
        pv, sv = [], []
        for r in sorted(Path(rd).glob("*.png")):
            gp = Path(gd) / r.name
            if not gp.exists(): continue
            p = TF.to_tensor(Image.open(r).convert("RGB")).unsqueeze(0).to(DEVICE)
            g = TF.to_tensor(Image.open(gp).convert("RGB")).unsqueeze(0).to(DEVICE)
            pv.append(compute_psnr(p, g))
            sv.append(compute_ssim(p, g))
        if pv:
            return {
                "PSNR" : round(sum(pv)/len(pv), 3),
                "SSIM" : round(sum(sv)/len(sv), 4),
                "N"    : len(pv),
                "split": split,
            }
    return {}

print("\nUtilities ready.")


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DUSt3R MODEL
# ═══════════════════════════════════════════════════════════════════════════════

if not os.path.isfile(DUST3R_CKPT):
    raise FileNotFoundError(f"DUSt3R weights not found: {DUST3R_CKPT}")

print(f"\nLoading DUSt3R from {DUST3R_CKPT}...")
t0 = time.time()
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images as dust3r_load
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_CKPT).to(DEVICE)
dust3r_model.eval()
print(f"Loaded in {time.time()-t0:.1f}s on {DEVICE}.")


# ═══════════════════════════════════════════════════════════════════════════════
#  DUSt3R INFERENCE + Sim3 GLOBAL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_dust3r(image_paths, img_size=512, niter=300):
    n = len(image_paths)
    # swin-5 for larger sets to avoid OOM; complete for very small sets
    scene_graph = "complete" if n <= 15 else "swin-5"
    print(f"  Loading {n} images @ {img_size}px  (scene_graph={scene_graph})")
    images = dust3r_load(image_paths, size=img_size)

    pairs = make_pairs(images, scene_graph=scene_graph,
                       prefilter=None, symmetrize=True)
    print(f"  {len(pairs)} pairs")

    # Forward pass on GPU
    print("  DUSt3R inference (GPU)...")
    t0 = time.time()
    output = inference(pairs, dust3r_model, DEVICE, batch_size=1, verbose=False)
    print(f"  → {time.time()-t0:.0f}s")

    # Global alignment on CPU to avoid OOM
    print("  Global Sim3 alignment (CPU)...")
    t0 = time.time()
    output_cpu = {}
    for k, v in output.items():
        if isinstance(v, list):
            output_cpu[k] = [
                {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv
                 for kk, vv in item.items()}
                if isinstance(item, dict) else item
                for item in v]
        elif isinstance(v, torch.Tensor):
            output_cpu[k] = v.cpu()
        else:
            output_cpu[k] = v

    scene = global_aligner(output_cpu, device="cpu",
                           mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule="cosine", lr=0.01)
    print(f"  → {time.time()-t0:.0f}s  loss={loss:.4f}")

    poses  = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pts_l  = scene.get_pts3d()
    msks   = scene.get_masks()

    all_pts, all_col = [], []
    for pts, msk, ip in zip(pts_l, msks, image_paths):
        p = pts.detach().cpu().numpy()
        m = msk.detach().cpu().numpy().astype(bool)
        img = np.array(Image.open(ip).convert("RGB").resize(
            (p.shape[1], p.shape[0])))
        all_pts.append(p[m]); all_col.append(img[m])

    pts3d = np.concatenate(all_pts, 0)
    col3d = np.concatenate(all_col, 0)
    print(f"  poses={poses.shape}  pts={pts3d.shape}")
    return poses, pts3d, col3d, focals


def save_as_colmap(out_dir, image_paths, poses_c2w, pts3d, col3d, focals):
    sp = f"{out_dir}/sparse/0"
    os.makedirs(sp, exist_ok=True)
    img0 = Image.open(image_paths[0]); W, H = img0.size
    fx = fy = float(np.mean(focals)); cx, cy = W/2.0, H/2.0
    write_cameras_bin(f"{sp}/cameras.bin", W, H, fx, fy, cx, cy)
    frames = []
    for ip, c2w in zip(image_paths, poses_c2w):
        w2c = np.linalg.inv(c2w)
        frames.append({"name": os.path.basename(ip),
                       "qvec": rotmat_to_quat(w2c[:3, :3]),
                       "tvec": w2c[:3, 3]})
    write_images_bin(f"{sp}/images.bin", frames)
    MAX = 500_000
    if len(pts3d) > MAX:
        idx = np.random.choice(len(pts3d), MAX, replace=False)
        pts3d, col3d = pts3d[idx], col3d[idx]
    write_points3d_bin(f"{sp}/points3D.bin", list(zip(pts3d, col3d)))
    link = f"{out_dir}/images"
    src  = os.path.dirname(image_paths[0])
    if not os.path.exists(link):
        os.symlink(src, link)
    print(f"  Saved → {sp}  ({len(frames)} poses, {len(pts3d)} pts)")
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  ATE (Absolute Trajectory Error)
# ═══════════════════════════════════════════════════════════════════════════════

def read_colmap_positions(images_bin):
    pos = []
    if not os.path.isfile(images_bin):
        print(f"  GT not found: {images_bin}"); return np.zeros((0,3))
    try:
        with open(images_bin, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100000: return np.zeros((0,3))
            for _ in range(n):
                f.read(4)
                qvec = struct.unpack("<dddd", f.read(32))
                tvec = np.array(struct.unpack("<ddd", f.read(24)))
                f.read(4)
                while True:
                    c = f.read(1)
                    if not c or c == b"\x00": break
                tl_bytes = f.read(8)
                if len(tl_bytes) < 8: break
                tl = struct.unpack("<Q", tl_bytes)[0]
                if tl > 1_000_000: return np.zeros((0,3))
                f.read(24 * tl)   # 24 bytes per 2D point
                w,x,y,z = qvec
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                    [2*(x*y+w*z),   1-2*(x*x+z*z),  2*(y*z-w*x)],
                    [2*(x*z-w*y),   2*(y*z+w*x),    1-2*(x*x+y*y)]])
                pos.append(-R.T @ tvec)
    except Exception as e:
        print(f"  WARNING: {e}"); return np.zeros((0,3))
    return np.array(pos) if pos else np.zeros((0,3))

def umeyama_align(P, Q):
    mu_P=P.mean(0); mu_Q=Q.mean(0)
    Pc=P-mu_P; Qc=Q-mu_Q
    sig2=(Pc**2).sum()/len(P)
    cov=(Qc.T@Pc)/len(P)
    U,D,Vt=np.linalg.svd(cov)
    S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; s=(D*S.diagonal()).sum()/sig2
    t=mu_Q-s*R@mu_P
    return (s*(R@Pc.T).T)+mu_Q, s

def compute_ate(est_poses_c2w, gt_bin):
    gt_pos  = read_colmap_positions(gt_bin)
    if len(gt_pos) == 0: return None
    est_pos = est_poses_c2w[:, :3, 3]
    n = min(len(gt_pos), len(est_pos))
    aligned, s = umeyama_align(est_pos[:n], gt_pos[:n])
    ate = float(np.sqrt(((aligned - gt_pos[:n])**2).sum(-1).mean()))
    print(f"  ATE RMSE={ate:.4f}m  scale={s:.3f}  N={n}")
    return ate


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE PER DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(name, cfg):
    print(f"\n{'='*60}\n  {name.upper()}  "
          f"(1/{cfg['step']} frames, ~{cfg['total']//cfg['step']} sparse)\n{'='*60}")

    out = f"{OUTPUT_ROOT}/{name}"
    os.makedirs(out, exist_ok=True)

    # 1. Subsample images — NO poses
    image_paths = list_images(cfg["images"], step=cfg["step"])
    total       = len(list_images(cfg["images"]))
    print(f"[1/5] {len(image_paths)} sparse images  (from {total} total)")
    if len(image_paths) < 3:
        print("  Too few images."); return {}

    # 2. DUSt3R — estimate poses from scratch
    poses_c2w   = None
    done_marker = f"{out}/sparse/0/cameras.bin"

    if os.path.isfile(done_marker):
        print("[2/5] DUSt3R already done — loading saved poses.")
        npy = f"{out}/poses_c2w.npy"
        if os.path.isfile(npy):
            poses_c2w = np.load(npy)
    else:
        print("[2/5] Running DUSt3R (no poses provided)...")
        poses_c2w, pts3d, col3d, focals = run_dust3r(
            image_paths,
            img_size=args.dust3r_img_size,
            niter=args.dust3r_niter)

        # 3. Save as COLMAP format for 3DGS
        print("[3/5] Saving COLMAP format...")
        save_as_colmap(out, image_paths, poses_c2w, pts3d, col3d, focals)
        np.save(f"{out}/poses_c2w.npy", poses_c2w)

    # 4. 3DGS training
    model   = f"{out}/3dgs_model"
    trained = os.path.isdir(f"{model}/point_cloud/iteration_{args.iterations}")

    if trained:
        print(f"[4/5] Already trained → {model}")
    else:
        train_py = (f"{REGGS_REPO}/train.py"
                    if os.path.isfile(f"{REGGS_REPO}/train.py")
                    else f"{GS_REPO}/train.py")
        print(f"[4/5] Training 3DGS ({os.path.basename(os.path.dirname(train_py))})...")
        t0 = time.time()
        r  = subprocess.run(
            f"python {train_py} "
            f"--source_path {out} "
            f"--model_path  {model} "
            f"--iterations  {args.iterations} --eval",
            shell=True)
        if r.returncode != 0:
            print("  WARNING: training failed."); return {}
        print(f"  Done in {(time.time()-t0)/60:.1f} min")

    # 5. Render
    render_dir = f"{model}/test/ours_{args.iterations}/renders"
    train_render = f"{model}/train/ours_{args.iterations}/renders"
    if not os.path.isdir(render_dir) and not os.path.isdir(train_render):
        print("[5/5] Rendering...")
        subprocess.run(
            f"python {GS_REPO}/render.py --model_path {model} --quiet",
            shell=True)
    else:
        print("[5/5] Renders already exist.")

    # Metrics
    print("\n--- Metrics ---")
    m = evaluate_model(model)
    if m:
        print(f"  PSNR={m['PSNR']}  SSIM={m['SSIM']}  "
              f"N={m['N']} ({m['split']} split)")

    # ATE
    if poses_c2w is None:
        npy = f"{out}/poses_c2w.npy"
        if os.path.isfile(npy): poses_c2w = np.load(npy)

    ate = None
    if poses_c2w is not None:
        print("Computing ATE vs COLMAP GT...")
        ate = compute_ate(poses_c2w, cfg["gt_poses"])

    result = {"dataset": name, **m}
    if ate is not None: result["ATE_RMSE"] = round(ate, 4)
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out}/metrics.json")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

all_results = {}
for name, cfg in DATASETS.items():
    if not os.path.isdir(cfg["images"]):
        print(f"\n[SKIP] {name}: {cfg['images']} not found"); continue
    r = run_pipeline(name, cfg)
    if r: all_results[name] = r


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  SUMMARY — Part 2 (Original Data, Correct Sparsity)")
print("="*60)

# Load Part 1 Plan A metrics for comparison
p1_metrics = {}
p1f = f"{WORKPLACE}/output/part1/metrics.json"
if os.path.exists(p1f):
    p1_metrics = json.load(open(p1f))

try:
    import pandas as pd
    rows = []
    sparse_info = {"waymo": "1/10 (19 frames)", "dl3dv": "1/30 (10 frames)",
                   "re10k": "1/30 (9 frames)"}
    for ds in DATASETS:
        mf = f"{OUTPUT_ROOT}/{ds}/metrics.json"
        if not os.path.exists(mf): continue
        m2 = json.load(open(mf))
        p1 = p1_metrics.get(f"{ds}_planA", {})
        delta = round(m2["PSNR"] - p1["PSNR"], 3) if p1.get("PSNR") and m2.get("PSNR") else "—"
        rows.append({
            "Dataset"          : ds,
            "Sparse ratio"     : sparse_info.get(ds, "—"),
            "P1 PSNR (dense)"  : p1.get("PSNR", "—"),
            "P1 SSIM (dense)"  : p1.get("SSIM", "—"),
            "P2 PSNR (sparse)" : m2.get("PSNR", "—"),
            "P2 SSIM (sparse)" : m2.get("SSIM", "—"),
            "ATE RMSE (m)"     : m2.get("ATE_RMSE", "—"),
            "ΔPSNR (P2-P1)"    : delta,
            "Eval split"       : m2.get("split", "—"),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    csv_path = f"{OUTPUT_ROOT}/part2_original_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
except ImportError:
    for name, r in all_results.items():
        print(f"  {name}: PSNR={r.get('PSNR','—')}  "
              f"SSIM={r.get('SSIM','—')}  ATE={r.get('ATE_RMSE','—')}")

# Trajectory plots (headless)
print("\nGenerating trajectory plots...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for name in all_results:
    pf = f"{OUTPUT_ROOT}/{name}/poses_c2w.npy"
    if not os.path.isfile(pf): continue
    poses = np.load(pf); cam = poses[:, :3, 3]
    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(cam[:,0], cam[:,1], cam[:,2], c="red", s=60, label="Est. cameras")
    for c2w in poses:
        pos = c2w[:3, 3]; fwd = c2w[:3, 2] * 0.3
        ax.quiver(*pos, *fwd, color="blue", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{name} — DUSt3R recovered cameras\n"
                 f"({DATASETS[name]['total']//DATASETS[name]['step']} sparse views, "
                 f"1/{DATASETS[name]['step']} sparsity)", fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(); plt.tight_layout()
    sp = f"{OUTPUT_ROOT}/figures/{name}_trajectory.png"
    plt.savefig(sp, dpi=150); plt.close()
    print(f"  Saved → {sp}")

print("\n" + "="*60)
print(f"  Part 2 COMPLETE  →  {OUTPUT_ROOT}")
print("="*60)