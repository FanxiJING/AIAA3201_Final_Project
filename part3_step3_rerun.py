#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Project 4 — Part 2 Alternative: RegGS-style / InstantSplat Unposed Sparse Reconstruction
Fixed version: removes the invalid init="known_poses" call that caused
"not all poses are known" error. The two-phase RegGS alignment now uses
init="mst" for both phases (phase 2 warm-starts from the scene object
that already holds phase-1 pose parameters, without re-initialising).

Existing outputs (sparse/0/, poses_c2w.npy, 3DGS models) are NEVER
overwritten — the script detects them and skips straight to evaluation,
so already-completed InstantSplat results are fully preserved.

Usage:
    conda activate dust3r
    python part3_alt.py [--datasets waymo dl3dv re10k]
                                [--method reggs|instantsplat|both]
                                [--iterations 30000]
                                [--dust3r_img_size 512]
                                [--dust3r_niter 300]
                                [--reg_niter 500]
'''

import os, sys, json, struct, time, subprocess, argparse
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Designate which gpu to use


# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--project4",        default="/data2/fjing221/project4")
parser.add_argument("--workplace",       default="/data2/fjing221/workplace")
parser.add_argument("--data_root",       default="/data2/fjing221/data")
parser.add_argument("--datasets", nargs="+",
                    default=["waymo", "dl3dv", "re10k"],
                    choices=["waymo", "dl3dv", "re10k"])
parser.add_argument("--method",          default="both",
                    choices=["reggs", "instantsplat", "both"])
parser.add_argument("--iterations",      type=int, default=30000)
parser.add_argument("--dust3r_img_size", type=int, default=512)
parser.add_argument("--dust3r_niter",    type=int, default=300)
parser.add_argument("--reg_niter",       type=int, default=500,
                    help="RegGS total alignment iterations (split 1/4 warm + 3/4 refine)")
parser.add_argument("--pseudo_views_dir", type=str, default=None,
                    help="Directory containing Difix-generated pseudo-views")
parser.add_argument("--use_pseudo_views", action="store_true",
                    help="Enable pseudo-views integration")
args = parser.parse_args()
if args.pseudo_views_dir and not args.use_pseudo_views:
    args.use_pseudo_views = True

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT4    = args.project4
WORKPLACE   = args.workplace
DATA_ROOT   = args.data_root
WEIGHTS_DIR = f"{WORKPLACE}/weights"
OUTPUT_ROOT = f"{WORKPLACE}/output/part3_alternative"
BASELINE    = f"{WORKPLACE}/output/part3_original"

GS_REPO     = f"{WORKPLACE}/gaussian-splatting"
DUST3R_REPO = f"{WORKPLACE}/dust3r"
REGGS_REPO  = f"{WORKPLACE}/RegGS"
DUST3R_CKPT = f"{WEIGHTS_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

DATASETS_ALL = {
    "waymo": {
        "images"  : f"{PROJECT4}/405841/FRONT/rgb",
        "step"    : 10,
        "gt_poses": f"{DATA_ROOT}/405841/dense/0/sparse/0/images.bin",
        "total"   : 199,
    },
    "dl3dv": {
        "images"  : f"{PROJECT4}/DL3DV-2/rgb",
        "step"    : 30,
        "gt_poses": f"{DATA_ROOT}/DL3DV-2/dense/0/sparse/0/images.bin",
        "total"   : 306,
    },
    "re10k": {
        "images"  : f"{PROJECT4}/Re10k-1/images",
        "step"    : 30,
        "gt_poses": f"{DATA_ROOT}/Rek10-v1/dense/0/sparse/0/images.bin",
        "total"   : 280,
    },
}
DATASETS = {k: v for k, v in DATASETS_ALL.items() if k in args.datasets}

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/figures", exist_ok=True)

for p in [GS_REPO, DUST3R_REPO, f"{DUST3R_REPO}/croco", REGGS_REPO]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ── Environment ───────────────────────────────────────────────────────────────
print("=" * 70)
print("  Part 2 Alternative (fixed) — RegGS / InstantSplat")
print("=" * 70)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}  "
      f"GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'}")
print(f"Method : {args.method}   Output: {OUTPUT_ROOT}\n")

IMG_EXTS = {".jpg", ".jpeg", ".png"}

print("Dataset image dirs:")
for name, cfg in DATASETS.items():
    exists = os.path.isdir(cfg["images"])
    if exists:
        n = len([f for f in os.listdir(cfg["images"])
                 if Path(f).suffix.lower() in IMG_EXTS
                 and not os.path.isdir(os.path.join(cfg["images"], f))])
    else:
        n = 0
    print(f"  {name:8s}: {n:4d} total -> {n // cfg['step']:3d} sparse "
          f"(1/{cfg['step']})  {'OK' if exists else 'MISSING'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def find_pseudo_views(pseudo_dir, original_images):
    if not os.path.exists(pseudo_dir):
        return []
    pseudo_files = [os.path.join(pseudo_dir, f) for f in os.listdir(pseudo_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return pseudo_files

def list_images(img_dir, step=1):
    files = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if not os.path.isdir(os.path.join(img_dir, f))
        and Path(f).suffix.lower() in IMG_EXTS
    ])
    return files[::step]

def rotmat_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s,
                         (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1, 1] > R[2, 2]:
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
        for v in [fx, fy, cx, cy]:
            f.write(struct.pack("<d", float(v)))

def write_images_bin(path, frames):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for i, fr in enumerate(frames):
            f.write(struct.pack("<I", i + 1))
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
                f.write(struct.pack("<Q", i + 1))
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
    m1  = F.avg_pool2d(p, k, 1, k // 2)
    m2  = F.avg_pool2d(g, k, 1, k // 2)
    s1  = F.avg_pool2d(p * p, k, 1, k // 2) - m1 ** 2
    s2  = F.avg_pool2d(g * g, k, 1, k // 2) - m2 ** 2
    s12 = F.avg_pool2d(p * g, k, 1, k // 2) - m1 * m2
    return (((2*m1*m2+C1)*(2*s12+C2)) /
            ((m1**2+m2**2+C1)*(s1+s2+C2))).mean().item()

def evaluate_model(model_path):
    for split in ["test", "train"]:
        rd = f"{model_path}/{split}/ours_{args.iterations}/renders"
        gd = f"{model_path}/{split}/ours_{args.iterations}/gt"
        if not os.path.isdir(rd):
            continue
        pv, sv = [], []
        for r in sorted(Path(rd).glob("*.png")):
            gp = Path(gd) / r.name
            if not gp.exists():
                continue
            p = TF.to_tensor(Image.open(r).convert("RGB")).unsqueeze(0).to(DEVICE)
            g = TF.to_tensor(Image.open(gp).convert("RGB")).unsqueeze(0).to(DEVICE)
            pv.append(compute_psnr(p, g))
            sv.append(compute_ssim(p, g))
        if pv:
            return {"PSNR": round(sum(pv)/len(pv), 3),
                    "SSIM": round(sum(sv)/len(sv), 4),
                    "N":    len(pv), "split": split}
    return {}

def read_colmap_positions(images_bin):
    pos = []
    if not os.path.isfile(images_bin):
        print(f"  GT not found: {images_bin}")
        return np.zeros((0, 3))
    try:
        with open(images_bin, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100_000:
                return np.zeros((0, 3))
            for _ in range(n):
                f.read(4)
                qvec = struct.unpack("<dddd", f.read(32))
                tvec = np.array(struct.unpack("<ddd", f.read(24)))
                f.read(4)
                while True:
                    c = f.read(1)
                    if not c or c == b"\x00":
                        break
                tl_bytes = f.read(8)
                if len(tl_bytes) < 8:
                    break
                tl = struct.unpack("<Q", tl_bytes)[0]
                if tl > 1_000_000:
                    return np.zeros((0, 3))
                f.read(24 * tl)
                w, x, y, z = qvec
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                    [2*(x*y+w*z),   1-2*(x*x+z*z),  2*(y*z-w*x)],
                    [2*(x*z-w*y),   2*(y*z+w*x),    1-2*(x*x+y*y)]])
                pos.append(-R.T @ tvec)
    except Exception as e:
        print(f"  WARNING reading GT: {e}")
        return np.zeros((0, 3))
    return np.array(pos) if pos else np.zeros((0, 3))

def umeyama_align(P, Q):
    mu_P = P.mean(0); mu_Q = Q.mean(0)
    Pc = P - mu_P;    Qc = Q - mu_Q
    sig2 = (Pc**2).sum() / len(P)
    cov  = (Qc.T @ Pc) / len(P)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / sig2
    t = mu_Q - s * R @ mu_P
    return (s * (R @ Pc.T).T) + mu_Q, s

def compute_ate(est_poses_c2w, gt_bin):
    gt_pos = read_colmap_positions(gt_bin)
    if len(gt_pos) == 0:
        return None
    est_pos = est_poses_c2w[:, :3, 3]
    n = min(len(gt_pos), len(est_pos))
    aligned, s = umeyama_align(est_pos[:n], gt_pos[:n])
    ate = float(np.sqrt(((aligned - gt_pos[:n])**2).sum(-1).mean()))
    print(f"  ATE RMSE={ate:.4f}m  scale={s:.3f}  N={n}")
    return ate

def train_3dgs(source_path, model_path):
    done = f"{model_path}/point_cloud/iteration_{args.iterations}"
    if os.path.isdir(done):
        print(f"  Already trained -> {model_path}")
        return True
    os.makedirs(model_path, exist_ok=True)
    cmd = (f"python {GS_REPO}/train.py "
           f"--source_path {source_path} "
           f"--model_path  {model_path} "
           f"--iterations  {args.iterations} --eval")
    print(f"  Training 3DGS -> {model_path}")
    t0 = time.time()
    r  = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print("  WARNING: 3DGS training failed.")
        return False
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    return True

def render_model(model_path):
    rd = f"{model_path}/test/ours_{args.iterations}/renders"
    tr = f"{model_path}/train/ours_{args.iterations}/renders"
    if os.path.isdir(rd) or os.path.isdir(tr):
        print("  Renders already exist.")
        return
    subprocess.run(
        f"python {GS_REPO}/render.py --model_path {model_path} --quiet",
        shell=True)

def save_as_colmap(out_dir, image_paths, poses_c2w, pts3d, col3d, focals, max_pts=500_000):
    sp = f"{out_dir}/sparse/0"
    os.makedirs(sp, exist_ok=True)
    
    # Create images directory (not symlink)
    img_dir = f"{out_dir}/images"
    os.makedirs(img_dir, exist_ok=True)
    
    # Copy or symlink all images to img_dir with consistent naming
    img_name_map = {}  # original path -> new name
    for i, img_path in enumerate(image_paths):
        ext = os.path.splitext(img_path)[1]
        new_name = f"{i:05d}{ext}"
        new_path = os.path.join(img_dir, new_name)
        if not os.path.exists(new_path):
            os.symlink(os.path.abspath(img_path), new_path)
        img_name_map[img_path] = new_name
    
    # Write cameras.bin
    img0 = Image.open(image_paths[0]); W, H = img0.size
    fx = fy = float(np.mean(focals)); cx, cy = W / 2.0, H / 2.0
    write_cameras_bin(f"{sp}/cameras.bin", W, H, fx, fy, cx, cy)
    
    # Write images.bin using new filenames
    frames = []
    for ip, c2w in zip(image_paths, poses_c2w):
        w2c = np.linalg.inv(c2w)
        frames.append({
            "name": img_name_map[ip],
            "qvec": rotmat_to_quat(w2c[:3, :3]),
            "tvec": w2c[:3, 3]
        })
    write_images_bin(f"{sp}/images.bin", frames)
    
    # Write points3d.bin
    if len(pts3d) > max_pts:
        idx = np.random.choice(len(pts3d), max_pts, replace=False)
        pts3d, col3d = pts3d[idx], col3d[idx]
    write_points3d_bin(f"{sp}/points3D.bin", list(zip(pts3d, col3d)))
    
    print(f"  Saved COLMAP -> {sp}  ({len(frames)} poses, {len(pts3d)} pts)")
    return out_dir

def _move_to_cpu(output):
    out = {}
    for k, v in output.items():
        if isinstance(v, list):
            out[k] = [
                {kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv)
                 for kk, vv in item.items()}
                if isinstance(item, dict) else item
                for item in v
            ]
        elif isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DUSt3R
# ═══════════════════════════════════════════════════════════════════════════════

if not os.path.isfile(DUST3R_CKPT):
    raise FileNotFoundError(f"DUSt3R weights not found: {DUST3R_CKPT}")

print(f"Loading DUSt3R from {DUST3R_CKPT}...")
t0 = time.time()
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images as dust3r_load
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_CKPT).to(DEVICE)
dust3r_model.eval()
print(f"Loaded in {time.time()-t0:.1f}s on {DEVICE}.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION A: RegGS-STYLE TWO-PHASE Sim(3) ALIGNMENT   [BUG FIXED]
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Root cause of "not all poses are known":
#    compute_global_alignment(init="known_poses") requires ALL camera poses to
#    be pre-supplied as fixed tensors via scene.preset_pose(). After an MST run
#    they exist internally but are NOT flagged as "known" in the DUSt3R API,
#    so the check raises the error.
#
#  Fix: call compute_global_alignment(init="mst") for BOTH phases on the SAME
#  scene object. On the second call, the optimizer's internal pose tensors
#  already hold the phase-1 solution; "mst" simply means "initialise if not yet
#  done" — since they are already initialised, the optimiser continues from
#  the current state with the new LR/schedule. No external pose injection needed.

def run_reggs_registration(image_paths, img_size=512, niter_ba=500):
    """
    RegGS-style two-phase Sim(3) alignment (fixed).

    Phase 1 (niter_ba//4 iters, linear, lr=0.05): MST warm-start.
    Phase 2 (niter_ba iters,     cosine, lr=0.01): refinement on same scene object.

    Both calls use init='mst'; the second call continues from phase-1 parameters
    because the scene object already holds them — no init="known_poses" needed.
    """
    n = len(image_paths)
    scene_graph = "complete" if n <= 15 else "swin-5"
    print(f"  [RegGS] {n} images @ {img_size}px  graph={scene_graph}")

    images = dust3r_load(image_paths, size=img_size)
    pairs  = make_pairs(images, scene_graph=scene_graph,
                        prefilter=None, symmetrize=True)
    print(f"  [RegGS] {len(pairs)} pairs — DUSt3R inference...")
    t0 = time.time()
    output = inference(pairs, dust3r_model, DEVICE, batch_size=1, verbose=False)
    print(f"  [RegGS] inference done in {time.time()-t0:.0f}s")

    niter_p1 = max(100, niter_ba // 4)
    niter_p2 = niter_ba

    print(f"  [RegGS] Phase 1: MST init, {niter_p1} iters, lr=0.05 (linear)...")
    t0 = time.time()
    output_cpu = _move_to_cpu(output)
    scene  = global_aligner(output_cpu, device="cpu",
                            mode=GlobalAlignerMode.PointCloudOptimizer)
    # Phase 1 — standard MST initialisation + coarse optimisation
    loss1  = scene.compute_global_alignment(
        init="mst", niter=niter_p1, schedule="linear", lr=0.05)
    print(f"  [RegGS] Phase 1 done: loss={loss1:.4f}  ({time.time()-t0:.0f}s)")

    # Phase 2 — continue on the SAME scene object (poses already set from phase 1)
    # init="mst" is safe: the optimiser checks if poses are already initialised
    # and skips re-initialisation, continuing from the current parameter state.
    print(f"  [RegGS] Phase 2: refine, {niter_p2} iters, lr=0.01 (cosine)...")
    t0 = time.time()
    loss2  = scene.compute_global_alignment(
        init="mst", niter=niter_p2, schedule="cosine", lr=0.01)
    print(f"  [RegGS] Phase 2 done: loss={loss2:.4f}  ({time.time()-t0:.0f}s)")

    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    focals    = scene.get_focals().detach().cpu().numpy()
    pts_l     = scene.get_pts3d()
    msks      = scene.get_masks()

    # Scale normalisation: set mean inter-camera distance to 1
    cam_pos   = poses_c2w[:, :3, 3]
    mean_dist = (np.linalg.norm(cam_pos - cam_pos.mean(0), axis=1).mean()
                 if len(cam_pos) > 1 else 1.0)
    scale = (1.0 / mean_dist) if mean_dist > 1e-6 else 1.0
    if scale != 1.0:
        poses_c2w[:, :3, 3] *= scale
        print(f"  [RegGS] Scale normalised (x{scale:.4f})")

    all_pts, all_col = [], []
    for pts, msk, ip in zip(pts_l, msks, image_paths):
        p = pts.detach().cpu().numpy()
        m = msk.detach().cpu().numpy().astype(bool)
        img = np.array(Image.open(ip).convert("RGB").resize(
            (p.shape[1], p.shape[0])))
        all_pts.append(p[m])
        all_col.append(img[m])

    pts3d = np.concatenate(all_pts, 0) * scale
    col3d = np.concatenate(all_col, 0)
    print(f"  [RegGS] poses={poses_c2w.shape}  pts={pts3d.shape}")
    return poses_c2w, pts3d, col3d, focals


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION B: InstantSplat-STYLE CONFIDENCE-WEIGHTED INIT
# ═══════════════════════════════════════════════════════════════════════════════

def run_instantsplat(image_paths, img_size=512, niter=300,
                     gaussians_per_view=8000):
    """
    DUSt3R global alignment + confidence-weighted point sampling.
    Pose estimation is identical to DUSt3R-only; the improvement is in
    the quality/distribution of the seeding point cloud for 3DGS.
    """
    n = len(image_paths)
    scene_graph = "complete" if n <= 15 else "swin-5"
    print(f"  [ISplat] {n} images @ {img_size}px  graph={scene_graph}")

    images = dust3r_load(image_paths, size=img_size)
    pairs  = make_pairs(images, scene_graph=scene_graph,
                        prefilter=None, symmetrize=True)
    print(f"  [ISplat] {len(pairs)} pairs — inference...")
    t0 = time.time()
    output = inference(pairs, dust3r_model, DEVICE, batch_size=1, verbose=False)
    print(f"  [ISplat] inference done in {time.time()-t0:.0f}s")

    print(f"  [ISplat] Global alignment (niter={niter})...")
    t0 = time.time()
    output_cpu = _move_to_cpu(output)
    scene = global_aligner(output_cpu, device="cpu",
                           mode=GlobalAlignerMode.PointCloudOptimizer)
    loss  = scene.compute_global_alignment(
        init="mst", niter=niter, schedule="cosine", lr=0.01)
    print(f"  [ISplat] done in {time.time()-t0:.0f}s  loss={loss:.4f}")

    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    focals    = scene.get_focals().detach().cpu().numpy()
    pts_l     = scene.get_pts3d()
    msks      = scene.get_masks()

    all_pts, all_col = [], []
    for pts, msk, ip in zip(pts_l, msks, image_paths):
        p = pts.detach().cpu().numpy()
        m = msk.detach().cpu().numpy().astype(bool)
        img = np.array(Image.open(ip).convert("RGB").resize(
            (p.shape[1], p.shape[0])))
        valid_pts = p[m]; valid_col = img[m]
        # Confidence-weighted sampling: prefer points near median scene depth
        if len(valid_pts) > gaussians_per_view:
            depths  = valid_pts[:, 2]
            z_med   = np.median(depths)
            weights = 1.0 / (np.abs(depths - z_med) + 0.1)
            weights /= weights.sum()
            idx      = np.random.choice(len(valid_pts), gaussians_per_view,
                                        replace=False, p=weights)
            valid_pts = valid_pts[idx]
            valid_col = valid_col[idx]
        all_pts.append(valid_pts)
        all_col.append(valid_col)

    pts3d = np.concatenate(all_pts, 0)
    col3d = np.concatenate(all_col, 0)
    print(f"  [ISplat] poses={poses_c2w.shape}  pts={pts3d.shape}")
    return poses_c2w, pts3d, col3d, focals


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERIC PIPELINE RUNNER  (resume-safe at every stage)
# ═══════════════════════════════════════════════════════════════════════════════

def run_method(method_name, name, cfg, run_fn, run_kwargs):
    """
    Full pipeline: reconstruction -> COLMAP save -> 3DGS train -> render -> eval.
    Each stage is skipped if its outputs already exist on disk, so previously
    completed results (e.g. InstantSplat on dl3dv / re10k) are preserved.
    """
    print(f"\n{'─'*70}\n  {method_name.upper()} | {name.upper()}  "
          f"(1/{cfg['step']}, ~{cfg['total']//cfg['step']} frames)\n{'─'*70}")

    suffix = "_with_pseudo" if args.use_pseudo_views else ""
    out    = f"{OUTPUT_ROOT}/{name}/{method_name}{suffix}"
    model  = f"{out}/3dgs"
    npy_f  = f"{out}/poses_c2w.npy"
    done_m = f"{out}/sparse/0/cameras.bin"

    image_paths = list_images(cfg["images"], step=cfg["step"])
    if args.use_pseudo_views and args.pseudo_views_dir:
        pseudo_dir = f"{args.pseudo_views_dir}/{name}/{args.method}/pseudo_views"
        pseudo_paths = find_pseudo_views(pseudo_dir, image_paths)
        if pseudo_paths:
            pseudo_paths_renamed = []
            for p in pseudo_paths:
                dirname = os.path.dirname(p)
                filename = os.path.basename(p)
                new_filename = f"pseudo_{filename}"
                new_path = os.path.join(dirname, new_filename)
                # Create symlink with new name
                if not os.path.exists(new_path):
                    os.symlink(filename, new_path)
                pseudo_paths_renamed.append(new_path)
            
            image_paths = image_paths + pseudo_paths_renamed
            print(f"  Added {len(pseudo_paths)} pseudo-views")


    print(f"  Sparse images: {len(image_paths)}")
    if len(image_paths) < 3:
        print("  Too few images — skipping.")
        return {}

    # Stage 1: reconstruction
    poses_c2w = None
    if os.path.isfile(done_m) and os.path.isfile(npy_f):
        print(f"  [Stage 1] Init already complete — loading cached poses.")
        poses_c2w = np.load(npy_f)
    else:
        print(f"  [Stage 1] Running {method_name}...")
        t0 = time.time()
        try:
            poses_c2w, pts3d, col3d, focals = run_fn(image_paths, **run_kwargs)
        except Exception as e:
            print(f"  ERROR during {method_name}: {e}")
            import traceback; traceback.print_exc()
            return {}
        print(f"  {method_name} done in {(time.time()-t0)/60:.1f} min")
        os.makedirs(out, exist_ok=True)
        save_as_colmap(out, image_paths, poses_c2w, pts3d, col3d, focals)
        np.save(npy_f, poses_c2w)

    # Stage 2: 3DGS training
    print(f"  [Stage 2] 3DGS training...")
    if not train_3dgs(out, model):
        return {}

    # Stage 3: rendering
    print(f"  [Stage 3] Rendering...")
    render_model(model)

    # Stage 4: metrics
    print(f"  [Stage 4] Computing metrics...")
    m = evaluate_model(model)
    if m:
        print(f"  PSNR={m['PSNR']}  SSIM={m['SSIM']}  "
              f"N={m['N']} ({m['split']})")
    else:
        print("  No render output — metrics unavailable.")

    # Stage 5: ATE
    print(f"  [Stage 5] ATE vs. COLMAP GT...")
    if poses_c2w is None and os.path.isfile(npy_f):
        poses_c2w = np.load(npy_f)
    ate = None
    if poses_c2w is not None:
        # Only use original images for ATE (first N images)
        n_original = len(list_images(cfg["images"], step=cfg["step"]))
        original_poses = poses_c2w[:n_original]
        ate = compute_ate(original_poses, cfg["gt_poses"])

    result = {"dataset": name, "method": method_name, **m}
    if ate is not None:
        result["ATE_RMSE"] = round(ate, 4)

    os.makedirs(f"{OUTPUT_ROOT}/{name}", exist_ok=True)
    with open(f"{OUTPUT_ROOT}/{name}/{method_name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {OUTPUT_ROOT}/{name}/{method_name}_metrics.json")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL METHODS x DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

methods_to_run = []
if args.method in ("reggs", "both"):
    methods_to_run.append((
        "reggs", run_reggs_registration,
        {"img_size": args.dust3r_img_size, "niter_ba": args.reg_niter}
    ))
if args.method in ("instantsplat", "both"):
    methods_to_run.append((
        "instantsplat", run_instantsplat,
        {"img_size": args.dust3r_img_size, "niter": args.dust3r_niter}
    ))

all_results = {}
for name, cfg in DATASETS.items():
    if not os.path.isdir(cfg["images"]):
        print(f"\n[SKIP] {name}: {cfg['images']} not found")
        continue
    for method_name, run_fn, kwargs in methods_to_run:
        r = run_method(method_name, name, cfg, run_fn, kwargs)
        if r:
            all_results[f"{name}_{method_name}"] = r


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE  (always populated from both new results AND cached files)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  COMPARISON TABLE  (all available results)")
print("=" * 70)

# Hard-coded baseline from part2_original — always available as a fallback
BASELINE_HARD = {
    "waymo": {"PSNR": 14.556, "SSIM": 0.489,  "ATE_RMSE": 0.2914},
    "dl3dv": {"PSNR": 11.317, "SSIM": 0.2219, "ATE_RMSE": 0.8092},
    "re10k": {"PSNR":  9.393, "SSIM": 0.2181, "ATE_RMSE": 1.9138},
}
# Try to load from disk (overrides hard-coded values if present)
baseline_metrics = dict(BASELINE_HARD)
for name in DATASETS:
    bf = f"{BASELINE}/{name}/metrics.json"
    if os.path.isfile(bf):
        try:
            with open(bf) as f:
                d = json.load(f)
            baseline_metrics[name] = {k: d[k] for k in
                                      ["PSNR", "SSIM", "ATE_RMSE"] if k in d}
        except Exception:
            pass

# Load any cached alternative results not produced in this run
for name in DATASETS:
    for method_name, _, _ in methods_to_run:
        key = f"{name}_{method_name}"
        if key not in all_results:
            mf = f"{OUTPUT_ROOT}/{name}/{method_name}_metrics.json"
            if os.path.isfile(mf):
                try:
                    with open(mf) as f:
                        all_results[key] = json.load(f)
                    print(f"  Loaded cached result: {key}")
                except Exception:
                    pass

sparse_info = {"waymo": "1/10 (19 fr.)",
               "dl3dv": "1/30 (10 fr.)",
               "re10k": "1/30 (9 fr.)"}

def fmt(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)

rows = []
for name in list(DATASETS.keys()):
    bm = baseline_metrics.get(name, {})
    rows.append({
        "Dataset" : name,
        "Sparsity": sparse_info.get(name, ""),
        "Method"  : "DUSt3R-only (baseline)",
        "PSNR"    : fmt(bm.get("PSNR", "—")),
        "SSIM"    : fmt(bm.get("SSIM", "—"), 4),
        "ATE (m)" : fmt(bm.get("ATE_RMSE", "—"), 4),
    })
    for method_name, _, _ in methods_to_run:
        r = all_results.get(f"{name}_{method_name}", {})
        rows.append({
            "Dataset" : "",
            "Sparsity": "",
            "Method"  : f"{method_name} (ours)",
            "PSNR"    : fmt(r.get("PSNR", "—")),
            "SSIM"    : fmt(r.get("SSIM", "—"), 4),
            "ATE (m)" : fmt(r.get("ATE_RMSE", "—"), 4),
        })

try:
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    csv_path = f"{OUTPUT_ROOT}/comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
except ImportError:
    for r in rows:
        print("  " + "  ".join(str(v) for v in r.values()))

with open(f"{OUTPUT_ROOT}/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved -> {OUTPUT_ROOT}/all_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating trajectory plots...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

n_cols = 1 + len(methods_to_run)
colors = ["steelblue", "seagreen", "darkorange"]

for name, cfg in DATASETS.items():
    fig = plt.figure(figsize=(5 * n_cols, 5))
    plot_specs = [
        ("DUSt3R-only\n(baseline)",
         f"{BASELINE}/{name}/poses_c2w.npy",
         "tomato")
    ] + [
        (f"{m[0]}\n(ours)",
         f"{OUTPUT_ROOT}/{name}/{m[0]}/poses_c2w.npy",
         colors[i])
        for i, m in enumerate(methods_to_run)
    ]

    for idx, (title, npy_path, color) in enumerate(plot_specs):
        ax = fig.add_subplot(1, n_cols, idx + 1, projection="3d")
        if os.path.isfile(npy_path):
            poses = np.load(npy_path)
            cam   = poses[:, :3, 3]
            ax.scatter(cam[:, 0], cam[:, 1], cam[:, 2],
                       c=color, s=60, zorder=5)
            for c2w in poses:
                pos = c2w[:3, 3]; fwd = c2w[:3, 2] * 0.15
                ax.quiver(*pos, *fwd, color="navy", linewidth=0.7, alpha=0.5)
        else:
            ax.text2D(0.5, 0.5, "No data\n(not run / failed)",
                      ha="center", va="center",
                      transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(f"{title}\n({name})", fontsize=8)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    plt.suptitle(f"{name} — Recovered Camera Trajectories", fontsize=11)
    plt.tight_layout()
    sp = f"{OUTPUT_ROOT}/figures/{name}_trajectory_compare.png"
    plt.savefig(sp, dpi=150)
    plt.close()
    print(f"  Saved -> {sp}")

print("\n" + "=" * 70)
print(f"  Part 2 Alternative COMPLETE  ->  {OUTPUT_ROOT}")
print("=" * 70)
print("""
Next steps:
  1. Copy numbers from comparison_table.csv into Table 2 of the LaTeX report.
  2. Replace figure placeholders with renders from:
       output/part2_alternative/{dataset}/{method}/3dgs/
  3. Insert trajectory plots from:
       output/part2_alternative/figures/
""")