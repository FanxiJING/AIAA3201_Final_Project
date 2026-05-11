#!/usr/bin/env python3
"""
Project 4 — Part 1: High-Fidelity Reconstruction (Initialization Analysis)
Tailored for: conda env "dust3r" on /data2/fjing221

Directory layout assumed:
    /data2/fjing221/
        data/
            405841/FRONT/rgb/          ← Waymo images
            DL3DV-2/rgb/
            DL3DV-2/cameras.json
            DL3DV-2/intrinsics.json
            Rek10-v1/images/           ← note: "Rek10-v1" as shown in your tree
            Rek10-v1/cameras.json
            Rek10-v1/intrinsics.json
        workplace/
            gaussian-splatting/        ← already cloned
            dust3r/                    ← already cloned
            RegGS/                     ← cloned by setup script
            weights/
                DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
            output/
                part1/

Usage:
    conda activate dust3r
    python part1_linux.py [options]

Options:
    --datasets waymo dl3dv re10k   (default: all three)
    --iterations 30000             (default: 30000)
    --skip_planA                   skip COLMAP plan
    --skip_planB                   skip DUSt3R plan
    --dust3r_img_size 512
    --dust3r_niter 300
"""

import os, sys, json, struct, time, subprocess, warnings, argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_root",    default="/data2/fjing221/data")
parser.add_argument("--workplace",    default="/data2/fjing221/workplace")
parser.add_argument("--datasets", nargs="+",
                    default=["waymo", "dl3dv", "re10k"],
                    choices=["waymo", "dl3dv", "re10k"])
parser.add_argument("--iterations",   type=int, default=30000)
parser.add_argument("--skip_planA",   action="store_true")
parser.add_argument("--skip_planB",   action="store_true")
parser.add_argument("--dust3r_img_size", type=int, default=512)
parser.add_argument("--dust3r_niter",    type=int, default=300)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = args.data_root
WORKPLACE   = args.workplace
WEIGHTS_DIR = f"{WORKPLACE}/weights"
OUTPUT_ROOT = f"{WORKPLACE}/output/part1"

GS_REPO     = f"{WORKPLACE}/gaussian-splatting"
DUST3R_REPO = f"{WORKPLACE}/dust3r"
DUST3R_CKPT = f"{WEIGHTS_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# Dataset configs — all three datasets already have COLMAP output in sparse/0/
# Structure: data/{name}/images/  + data/{name}/sparse/0/{cameras,images,points3D}.bin
DATASETS_ALL = {
    "waymo": {
        "images"      : f"{DATA_ROOT}/405841/dense/0/images",
        "image_subdir": "images",
        "cameras_json": None,
        "intrinsics"  : None,
        "colmap_src"  : f"{DATA_ROOT}/405841/dense/0",
    },
    "dl3dv": {
        "images"      : f"{DATA_ROOT}/DL3DV-2/dense/0/images",
        "image_subdir": "images",
        "cameras_json": None,
        "intrinsics"  : None,
        "colmap_src"  : f"{DATA_ROOT}/DL3DV-2/dense/0",
    },
    "re10k": {
        "images"      : f"{DATA_ROOT}/Rek10-v1/dense/0/images",
        "image_subdir": "images",
        "cameras_json": None,
        "intrinsics"  : None,
        "colmap_src"  : f"{DATA_ROOT}/Rek10-v1/dense/0",
    },
}
DATASETS = {k: v for k, v in DATASETS_ALL.items() if k in args.datasets}

# Create output dirs
for d in [WEIGHTS_DIR, OUTPUT_ROOT, f"{OUTPUT_ROOT}/figures"]:
    os.makedirs(d, exist_ok=True)

# Add repos to Python path
for p in [GS_REPO, DUST3R_REPO, f"{DUST3R_REPO}/croco"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ── Environment check ─────────────────────────────────────────────────────────
print("=" * 60)
print("  Part 1 — Initialization Analysis")
print("=" * 60)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nPyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"Device   : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

print(f"\nWorkplace: {WORKPLACE}")
print(f"Data     : {DATA_ROOT}")
print(f"Datasets : {list(DATASETS.keys())}")
print(f"Iters    : {args.iterations}")

print("\nData sanity check:")
for name, cfg in DATASETS.items():
    exists = os.path.isdir(cfg["images"])
    n = 0
    if exists:
        n = len([f for f in os.listdir(cfg["images"])
                 if not os.path.isdir(os.path.join(cfg["images"], f))])
    print(f"  {name:8s}: {'OK' if exists else 'MISSING':7s} "
          f"({n} files) → {cfg['images']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def sh(cmd, check=True, capture=False):
    kw = dict(shell=True, text=True)
    if capture:
        kw.update(capture_output=True)
    r = subprocess.run(cmd, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")
    return r

def list_images(img_dir):
    return sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if not os.path.isdir(os.path.join(img_dir, f))
        and Path(f).suffix.lower() in IMG_EXTS
    ])

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
            f.write(struct.pack("<I", i + 1))
            f.write(struct.pack("<dddd", *fr["qvec"]))
            f.write(struct.pack("<ddd", *fr["tvec"]))
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

def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    for fxk in ["fx", "fl_x", "focal_x", "focal_length"]:
        if fxk in d:
            fx = float(d[fxk])
            fy = float(d.get("fy", d.get("fl_y", d.get("focal_y", fx))))
            cx = float(d.get("cx", d.get("principal_point_x", 0)))
            cy = float(d.get("cy", d.get("principal_point_y", 0)))
            w  = int(d.get("width",  d.get("w", d.get("image_width", 0))))
            h  = int(d.get("height", d.get("h", d.get("image_height", 0))))
            return fx, fy, cx, cy, w, h
    raise ValueError(f"Cannot parse intrinsics from {path}. Keys: {list(d.keys())}")

def load_cameras_json(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        frames = data
    elif "frames" in data:
        frames = data["frames"]
    elif all(isinstance(v, (list, dict)) for v in data.values()):
        frames = [{"file_path": k,
                   "transform_matrix": v if isinstance(v, list)
                   else v.get("transform_matrix")}
                  for k, v in data.items()]
    else:
        raise ValueError("Unknown cameras.json format")
    results = []
    for fr in frames:
        mat = None
        for k in ["transform_matrix", "c2w", "extrinsic", "pose"]:
            if k in fr:
                mat = np.array(fr[k], dtype=np.float64)
                break
        if mat is None:
            continue
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0, 0, 0, 1]])
        name = ""
        for k in ["file_path", "filename", "image", "image_name", "name"]:
            if k in fr and fr[k]:
                name = os.path.basename(str(fr[k]))
                break
        w2c = np.linalg.inv(mat)
        R, t = w2c[:3, :3], w2c[:3, 3]
        results.append({"name": name, "qvec": rotmat_to_quat(R), "tvec": t})
    return results

import torch.nn.functional as F
import torchvision.transforms.functional as TF

def compute_psnr(p, g):
    return (10 * torch.log10(torch.tensor(1.0) / F.mse_loss(p, g))).item()

def compute_ssim(p, g, k=11):
    C1, C2 = 0.01**2, 0.03**2
    m1 = F.avg_pool2d(p, k, 1, k//2)
    m2 = F.avg_pool2d(g, k, 1, k//2)
    s1  = F.avg_pool2d(p*p, k, 1, k//2) - m1**2
    s2  = F.avg_pool2d(g*g, k, 1, k//2) - m2**2
    s12 = F.avg_pool2d(p*g, k, 1, k//2) - m1*m2
    num = (2*m1*m2 + C1) * (2*s12 + C2)
    den = (m1**2 + m2**2 + C1) * (s1 + s2 + C2)
    return (num / den).mean().item()

def evaluate_model(model_path):
    iters = args.iterations
    rd = f"{model_path}/train/ours_{iters}/renders"
    gd = f"{model_path}/train/ours_{iters}/gt"
    if not os.path.isdir(rd):
        return {}
    pv, sv = [], []
    for r in sorted(Path(rd).glob("*.png")):
        gp = Path(gd) / r.name
        if not gp.exists():
            continue
        p = TF.to_tensor(Image.open(r).convert("RGB")).unsqueeze(0).to(DEVICE)
        g = TF.to_tensor(Image.open(gp).convert("RGB")).unsqueeze(0).to(DEVICE)
        pv.append(compute_psnr(p, g))
        sv.append(compute_ssim(p, g))
    if not pv:
        return {}
    return {
        "PSNR": round(sum(pv) / len(pv), 3),
        "SSIM": round(sum(sv) / len(sv), 4),
        "N":    len(pv)
    }

print("\nUtilities ready.")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Detect COLMAP flags (COLMAP 3.10)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_colmap_flags():
    """Version-safe flag detection for COLMAP 3.10."""
    fe = sh("colmap feature_extractor --help", capture=True, check=False).stderr
    fm = sh("colmap exhaustive_matcher  --help", capture=True, check=False).stderr

    def pick(h, opts):
        for o in opts:
            if o.lstrip("-") in h:
                return o
        return ""

    flags = {
        "single_cam" : pick(fe, ["--ImageReaderOptions.single_camera",
                                  "--ImageReader.single_camera",
                                  "--single_camera"]),
        "fe_gpu"     : pick(fe, ["--FeatureExtraction.use_gpu",
                                  "--SiftExtraction.use_gpu",
                                  "--use_gpu"]),
        "fe_maxfeat" : pick(fe, ["--FeatureExtraction.max_num_features",
                                  "--SiftExtraction.max_num_features",
                                  "--max_num_features"]),
        "fm_gpu"     : pick(fm, ["--FeatureMatching.use_gpu",
                                  "--SiftMatching.use_gpu",
                                  "--use_gpu"]),
    }
    print("COLMAP 3.10 flags:")
    for k, v in flags.items():
        print(f"  {k:12s}: {v or '(omitted)'}")
    return flags

try:
    r = sh("colmap --version", capture=True, check=False)
    ver_line = (r.stdout + r.stderr).strip().splitlines()[0]
    print(f"\nCOLMAP: {ver_line}")
    CFLAGS = detect_colmap_flags()
    HAVE_COLMAP = True
except Exception as e:
    print(f"WARNING: COLMAP unavailable: {e}")
    CFLAGS = {}
    HAVE_COLMAP = False


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Plan A: COLMAP pipeline
# ═══════════════════════════════════════════════════════════════════════════════

import sqlite3

def run_colmap(image_dir, out_dir, tag):
    """Full COLMAP SfM: extract → match → mapper → undistort. Resume-safe."""
    done = f"{out_dir}/dense/sparse/cameras.bin"
    if os.path.isfile(done):
        print(f"  [{tag}] COLMAP already done.")
        return f"{out_dir}/dense"

    db = f"{out_dir}/database.db"
    os.makedirs(f"{out_dir}/sparse", exist_ok=True)
    os.makedirs(f"{out_dir}/dense",  exist_ok=True)

    # Remove corrupt empty DB
    if os.path.isfile(db):
        try:
            con = sqlite3.connect(db)
            n = con.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
            con.close()
            if n == 0:
                os.remove(db)
            else:
                print(f"  [{tag}] Resuming — DB has {n} keypoints.")
        except:
            os.remove(db)

    USE_GPU = 1 if torch.cuda.is_available() else 0

    # Stage 1: feature extraction
    if not os.path.isfile(db):
        a = f"--database_path {db} --image_path {image_dir}"
        if CFLAGS.get("single_cam"):  a += f" {CFLAGS['single_cam']} 1"
        if CFLAGS.get("fe_gpu"):      a += f" {CFLAGS['fe_gpu']} {USE_GPU}"
        if CFLAGS.get("fe_maxfeat"):  a += f" {CFLAGS['fe_maxfeat']} 8192"
        print(f"  [{tag}] 1/4 feature_extractor...")
        t0 = time.time()
        sh(f"colmap feature_extractor {a}")
        print(f"        done in {time.time()-t0:.0f}s")
    else:
        print(f"  [{tag}] 1/4 feature_extractor — skipped (DB exists)")

    # Clear stale matches (prevents SQLite SIGABRT on resume)
    try:
        con = sqlite3.connect(db)
        con.execute("DELETE FROM matches")
        con.execute("DELETE FROM two_view_geometries")
        con.commit(); con.close()
    except:
        pass

    # Stage 2: matching
    n_imgs  = len(list_images(image_dir))
    matcher = "exhaustive_matcher" if n_imgs <= 300 else "vocab_tree_matcher"
    m_args  = f"--database_path {db}"
    if CFLAGS.get("fm_gpu"):
        m_args += f" {CFLAGS['fm_gpu']} {USE_GPU}"
    print(f"  [{tag}] 2/4 {matcher} ({n_imgs} images)...")
    t0 = time.time()
    sh(f"colmap {matcher} {m_args}")
    print(f"        done in {time.time()-t0:.0f}s")

    con = sqlite3.connect(db)
    nm = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]; con.close()
    print(f"        match pairs: {nm}")
    if nm == 0:
        raise RuntimeError(f"[{tag}] Zero feature matches — check images.")

    # Stage 3: mapper
    print(f"  [{tag}] 3/4 mapper...")
    t0 = time.time()
    sh(f"colmap mapper "
       f"--database_path {db} --image_path {image_dir} "
       f"--output_path {out_dir}/sparse "
       f"--Mapper.num_threads {os.cpu_count()} "
       f"--Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0")
    print(f"        done in {time.time()-t0:.0f}s")
    if not os.path.isdir(f"{out_dir}/sparse/0"):
        raise RuntimeError(f"[{tag}] Mapper produced no reconstruction.")

    # Stage 4: undistortion → produces dense/ with images + sparse subfolder
    print(f"  [{tag}] 4/4 image_undistorter...")
    t0 = time.time()
    sh(f"colmap image_undistorter "
       f"--image_path {image_dir} "
       f"--input_path {out_dir}/sparse/0 "
       f"--output_path {out_dir}/dense "
       f"--output_type COLMAP")
    print(f"        done in {time.time()-t0:.0f}s")
    print(f"  [{tag}] Done → {out_dir}/dense")
    return f"{out_dir}/dense"


def json_to_colmap(dataset_dir, image_subdir, out_dir):
    """Convert cameras.json + intrinsics.json to COLMAP binary format."""
    sp = f"{out_dir}/sparse/0"
    if os.path.isfile(f"{sp}/cameras.bin"):
        print(f"  COLMAP bins already exist: {sp}")
        return dataset_dir
    os.makedirs(sp, exist_ok=True)

    fx, fy, cx, cy, w, h = load_intrinsics(f"{dataset_dir}/intrinsics.json")
    if w == 0 or h == 0:
        imgs = list_images(f"{dataset_dir}/{image_subdir}")
        with Image.open(imgs[0]) as im:
            w, h = im.size
    if cx == 0: cx = w / 2.0
    if cy == 0: cy = h / 2.0
    print(f"  Intrinsics: {w}x{h}  fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}")

    frames = load_cameras_json(f"{dataset_dir}/cameras.json")
    print(f"  Loaded {len(frames)} poses from cameras.json")

    write_cameras_bin(f"{sp}/cameras.bin", w, h, fx, fy, cx, cy)
    write_images_bin( f"{sp}/images.bin",  frames)
    write_points3d_bin(f"{sp}/points3D.bin")   # empty — 3DGS will init from poses
    print(f"  Written COLMAP bins → {sp}")

    # 3DGS needs an 'images' symlink at the source_path level
    img_full = os.path.join(dataset_dir, image_subdir)
    link     = os.path.join(out_dir, "images")
    if not os.path.exists(link):
        os.symlink(img_full, link)
    return out_dir   # out_dir is the source_path for 3DGS


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Plan B: DUSt3R initialization
# ═══════════════════════════════════════════════════════════════════════════════

def load_dust3r():
    """Import DUSt3R — already in sys.path from the cloned repo."""
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images as d3r_load
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    return (AsymmetricCroCo3DStereo, inference, make_pairs,
            d3r_load, global_aligner, GlobalAlignerMode)

def run_dust3r_init(image_paths, model, img_size=512, niter=300):
    (_, inference, make_pairs,
     d3r_load, global_aligner, GlobalAlignerMode) = load_dust3r()

    print(f"  Loading {len(image_paths)} images at {img_size}px...")
    images = d3r_load(image_paths, size=img_size)

    n   = len(image_paths)
    win = None if n <= 50 else 10
    pairs = make_pairs(images,
                       scene_graph="complete" if win is None else "swin",
                       prefilter=None, symmetrize=True)
    print(f"  {len(pairs)} pairs  (window={'all' if win is None else win})")

    print("  DUSt3R inference...")
    t0 = time.time()
    output = inference(pairs, model, DEVICE, batch_size=1, verbose=False)
    print(f"  Inference: {time.time()-t0:.0f}s")

    print("  Global Sim3 alignment...")
    t0 = time.time()
    scene = global_aligner(output, device=DEVICE,
                           mode=GlobalAlignerMode.PointCloudOptimizer)
    loss  = scene.compute_global_alignment(
        init="mst", niter=niter, schedule="cosine", lr=0.01)
    print(f"  Alignment: {time.time()-t0:.0f}s  loss={loss:.4f}")

    poses  = scene.get_im_poses().detach().cpu().numpy()   # (N,4,4)
    focals = scene.get_focals().detach().cpu().numpy()     # (N,)
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


def save_dust3r_as_colmap(out_dir, image_paths, poses_c2w, pts3d, col3d, focals):
    """Save DUSt3R output in COLMAP sparse/0/ format for 3DGS train.py."""
    sp = f"{out_dir}/sparse/0"
    os.makedirs(sp, exist_ok=True)

    img0 = Image.open(image_paths[0]); W, H = img0.size
    fx = fy = float(np.mean(focals)); cx, cy = W / 2.0, H / 2.0
    write_cameras_bin(f"{sp}/cameras.bin", W, H, fx, fy, cx, cy)

    frames = []
    for ip, c2w in zip(image_paths, poses_c2w):
        w2c = np.linalg.inv(c2w)
        frames.append({
            "name": os.path.basename(ip),
            "qvec": rotmat_to_quat(w2c[:3, :3]),
            "tvec": w2c[:3, 3],
        })
    write_images_bin(f"{sp}/images.bin", frames)

    MAX_PTS = 500_000
    if len(pts3d) > MAX_PTS:
        idx = np.random.choice(len(pts3d), MAX_PTS, replace=False)
        pts3d, col3d = pts3d[idx], col3d[idx]
    write_points3d_bin(f"{sp}/points3D.bin", list(zip(pts3d, col3d)))

    # Symlink so 3DGS finds images under out_dir/images/
    link = f"{out_dir}/images"
    src  = os.path.dirname(image_paths[0])
    if not os.path.exists(link):
        os.symlink(src, link)

    print(f"  COLMAP bins → {sp}")
    print(f"  cameras=1  images={len(frames)}  pts={len(pts3d)}")
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — 3DGS training helper
# ═══════════════════════════════════════════════════════════════════════════════

def train_3dgs(source_path, model_path, images_subdir=None,
               iterations=30000):
    done = f"{model_path}/point_cloud/iteration_{iterations}"
    if os.path.isdir(done):
        print(f"  Already trained → {model_path}")
        return True
    os.makedirs(model_path, exist_ok=True)

    cmd = (f"python {GS_REPO}/train.py "
           f"--source_path {source_path} "
           f"--model_path  {model_path} "
           f"--iterations  {iterations} --eval")
    if images_subdir:
        cmd += f" --images {images_subdir}"

    print(f"  Training → {model_path}")
    t0 = time.time()
    r  = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print("  WARNING: 3DGS training failed — check logs."); return False
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

planA_source = {}
planB_source = {}

# ────────────────────────────────────────────────────────── Plan A: COLMAP ───
# All datasets already have COLMAP output — use them directly as source_path.
# 3DGS train.py needs: source_path/images/ + source_path/sparse/0/
# which matches exactly: /data2/fjing221/data/{name}/

if not args.skip_planA:
    print("\n" + "="*60)
    print("  PLAN A — Using pre-existing COLMAP output")
    print("="*60)

    for name, cfg in DATASETS.items():
        print(f"\n{'─'*55}\nPlan A | {name}\n{'─'*55}")

        if not os.path.isdir(cfg["images"]):
            print(f"  Images not found: {cfg['images']} — skipping")
            continue

        src = cfg["colmap_src"]
        sp  = f"{src}/sparse/0"

        # Verify required COLMAP bins exist
        missing = [f for f in ["cameras.bin", "images.bin", "points3D.bin"]
                   if not os.path.isfile(f"{sp}/{f}")]
        if missing:
            print(f"  WARNING: sparse/0 missing: {missing}")
            print(f"  Available: {os.listdir(sp)}")
            # points3D may be absent — create empty one so 3DGS doesn't crash
            if "points3D.bin" in missing:
                write_points3d_bin(f"{sp}/points3D.bin")
                print(f"  Created empty points3D.bin")
            missing = [f for f in ["cameras.bin", "images.bin"]
                       if not os.path.isfile(f"{sp}/{f}")]
            if missing:
                print(f"  FATAL: {missing} still missing — skipping")
                continue

        n_imgs = len(list_images(cfg["images"]))
        print(f"  images/     : {n_imgs} files")
        print(f"  sparse/0/   : {os.listdir(sp)}")
        print(f"  source_path : {src}")

        planA_source[name] = src

else:
    print("\nSkipping Plan A (--skip_planA).")

# ────────────────────────────────────────────────────────── Plan B: DUSt3R ───
if not args.skip_planB:
    print("\n" + "="*60)
    print("  PLAN B — DUSt3R Initialization")
    print("="*60)

    if not os.path.isfile(DUST3R_CKPT):
        raise FileNotFoundError(
            f"DUSt3R weights not found: {DUST3R_CKPT}\n"
            f"Run:  wget https://download.europe.naverlabs.com/ComputerVision/"
            f"DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -O {DUST3R_CKPT}")

    print(f"Loading DUSt3R from {DUST3R_CKPT}...")
    AsymmetricCroCo3DStereo = load_dust3r()[0]
    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_CKPT).to(DEVICE)
    dust3r_model.eval()
    print(f"Loaded on {DEVICE}.")

    for name, cfg in DATASETS.items():
        print(f"\n{'─'*55}\nPlan B | {name}\n{'─'*55}")
        out  = f"{OUTPUT_ROOT}/{name}/planB/dust3r_init"
        done = f"{out}/sparse/0/cameras.bin"

        if not os.path.isdir(cfg["images"]):
            print(f"  Images not found — skipping"); continue

        if os.path.isfile(done):
            print("  DUSt3R init already done.")
            planB_source[name] = out; continue

        try:
            image_paths = list_images(cfg["images"])
            print(f"  {len(image_paths)} images")
            poses, pts3d, col3d, focals = run_dust3r_init(
                image_paths, dust3r_model,
                img_size=args.dust3r_img_size,
                niter=args.dust3r_niter)
            save_dust3r_as_colmap(out, image_paths, poses, pts3d, col3d, focals)
            planB_source[name] = out
            print(f"  source_path = {out}")
        except Exception as e:
            print(f"  Plan B FAILED for {name}: {e}")
else:
    print("\nSkipping Plan B (--skip_planB).")

# ────────────────────────────────────────────────────── 3DGS Training ────────
print("\n" + "="*60)
print("  3DGS TRAINING")
print("="*60)

train_jobs = []
for name, cfg in DATASETS.items():
    if name in planA_source:
        train_jobs.append((
            f"{name}_planA",
            planA_source[name],
            f"{OUTPUT_ROOT}/{name}/planA/3dgs",
            None))   # images/ already at source_path/images/ — no subdir needed
    if name in planB_source:
        train_jobs.append((
            f"{name}_planB",
            planB_source[name],
            f"{OUTPUT_ROOT}/{name}/planB/3dgs",
            None))

for tag, src, mdl, imgs in train_jobs:
    print(f"\n{'─'*55}\n3DGS: {tag}\n{'─'*55}")
    train_3dgs(src, mdl, images_subdir=imgs, iterations=args.iterations)

# ────────────────────────────────────────────────────── Render & Evaluate ────
print("\n" + "="*60)
print("  RENDERING & EVALUATION")
print("="*60)

all_metrics = {}
for tag, _, mdl, _ in train_jobs:
    rd = f"{mdl}/train/ours_{args.iterations}/renders"
    if not os.path.isdir(rd):
        if os.path.isdir(f"{mdl}/point_cloud/iteration_{args.iterations}"):
            print(f"Rendering {tag}...")
            subprocess.run(
                f"python {GS_REPO}/render.py --model_path {mdl} --quiet",
                shell=True)
        else:
            print(f"  [{tag}] not trained — skip"); continue
    m = evaluate_model(mdl)
    all_metrics[tag] = m
    print(f"  {tag:22s}: PSNR={m.get('PSNR','—')}  "
          f"SSIM={m.get('SSIM','—')}  N={m.get('N','—')}")

with open(f"{OUTPUT_ROOT}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nMetrics saved → {OUTPUT_ROOT}/metrics.json")

# ────────────────────────────────────────────────────── Results table ─────────
print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
try:
    import pandas as pd
    rows = []
    for ds in DATASETS:
        mA = all_metrics.get(f"{ds}_planA", {})
        mB = all_metrics.get(f"{ds}_planB", {})
        dpsnr = (round(mB["PSNR"] - mA["PSNR"], 3)
                 if mA.get("PSNR") and mB.get("PSNR") else "—")
        rows.append({
            "Dataset"     : ds,
            "PlanA PSNR↑" : mA.get("PSNR", "—"),
            "PlanA SSIM↑" : mA.get("SSIM", "—"),
            "PlanB PSNR↑" : mB.get("PSNR", "—"),
            "PlanB SSIM↑" : mB.get("SSIM", "—"),
            "ΔPSNR (B−A)" : dpsnr,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    csv_path = f"{OUTPUT_ROOT}/part1_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
except ImportError:
    for row in rows:
        print(row)

# ────────────────────────────────────────────────────── Convergence plot ──────
print("\nGenerating plots (headless Agg backend)...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_tb_psnr(model_path):
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
        ea = EventAccumulator(model_path); ea.Reload()
        tags = ea.Tags().get("scalars", [])
        tag  = next((t for t in tags if "psnr" in t.lower()), None)
        if tag is None: return [], []
        ev = ea.Scalars(tag)
        return [e.step for e in ev], [e.value for e in ev]
    except:
        return [], []

n_ds = len(DATASETS)
fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4))
if n_ds == 1:
    axes = [axes]

for ax, ds in zip(axes, DATASETS.keys()):
    for plan, color, label in [
        ("planA", "steelblue", "Plan A (COLMAP)"),
        ("planB", "tomato",    "Plan B (DUSt3R)")
    ]:
        mdl    = f"{OUTPUT_ROOT}/{ds}/{plan}/3dgs"
        steps, vals = read_tb_psnr(mdl)
        if steps:
            ax.plot(steps, vals, color=color, label=label, linewidth=1.5)
        else:
            m = all_metrics.get(f"{ds}_{plan}", {})
            if m.get("PSNR"):
                ax.axhline(m["PSNR"], color=color, linestyle="--",
                           label=f"{label} (final {m['PSNR']:.1f} dB)")
    ax.set_title(ds, fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Plan A vs Plan B — Convergence", fontsize=13)
plt.tight_layout()
conv_path = f"{OUTPUT_ROOT}/figures/convergence.png"
plt.savefig(conv_path, dpi=150); plt.close()
print(f"Saved → {conv_path}")

print("\n" + "="*60)
print(f"  Part 1 COMPLETE  →  {OUTPUT_ROOT}")
print("="*60)
