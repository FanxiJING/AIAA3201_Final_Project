#!/usr/bin/env python3
"""
Step 1: Generate intermediate camera poses and render.
DO NOT import part2_alt.py to avoid argument conflicts.
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R



# ============================================================================
# Argument parsing (only Step 1 arguments)
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--workplace", default="/data2/fjing221/workplace")
parser.add_argument("--dataset", choices=["waymo", "dl3dv", "re10k"], required=True)
parser.add_argument("--method", choices=["reggs", "instantsplat"], required=True)
parser.add_argument("--num_interp", type=int, default=5)
parser.add_argument("--max_renders", type=int, default=-1)
parser.add_argument("--iterations", type=int, default=30000)

args = parser.parse_args()

print("=" * 70)
print("  Step 1: Generate Intermediate Poses and Render")
print(f"  Dataset: {args.dataset}, Method: {args.method}")
print(f"  Num interp: {args.num_interp}, Max renders: {args.max_renders}")
print("=" * 70)

# ============================================================================
# Paths
# ============================================================================
WORKPLACE = args.workplace
PART2_BASE = f"{WORKPLACE}/output/part2_alternative/{args.dataset}/{args.method}"
PART3_OUT = f"{WORKPLACE}/output/part3_intermediate/{args.dataset}/{args.method}"
MODEL_PATH = f"{PART2_BASE}/3dgs"
GS_REPO = f"{WORKPLACE}/gaussian-splatting"

os.makedirs(PART3_OUT, exist_ok=True)
os.makedirs(f"{PART3_OUT}/intermediate_renders", exist_ok=True)

print(f"  Part 2 base: {PART2_BASE}")
print(f"  Model path: {MODEL_PATH}")

# ============================================================================
# Load poses
# ============================================================================
npy_path = f"{PART2_BASE}/poses_c2w.npy"
if not os.path.exists(npy_path):
    raise FileNotFoundError(f"Poses not found: {npy_path}")

poses = np.load(npy_path)
print(f"  Loaded {len(poses)} camera poses")

# ============================================================================
# Interpolation function
# ============================================================================
def interpolate_poses_slerp(pose_A, pose_B, num_interp):
    t_A, t_B = pose_A[:3, 3], pose_B[:3, 3]
    rot_A = R.from_matrix(pose_A[:3, :3])
    rot_B = R.from_matrix(pose_B[:3, :3])
    
    # Convert to quaternions
    q_A = rot_A.as_quat()  # (x, y, z, w)
    q_B = rot_B.as_quat()
    
    interp_poses = []
    for i in range(1, num_interp + 1):
        alpha = i / (num_interp + 1)
        t_interp = (1 - alpha) * t_A + alpha * t_B
        
        # Spherical linear interpolation for quaternions
        dot = np.dot(q_A, q_B)
        # Ensure shortest path
        if dot < 0:
            q_B = -q_B
            dot = -dot
        # Clamp to avoid numerical issues
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        
        if theta < 1e-6:
            q_interp = q_A
        else:
            w1 = np.sin((1 - alpha) * theta) / np.sin(theta)
            w2 = np.sin(alpha * theta) / np.sin(theta)
            q_interp = w1 * q_A + w2 * q_B
        
        # Convert quaternion back to rotation matrix
        rot_interp = R.from_quat(q_interp)
        
        pose_interp = np.eye(4)
        pose_interp[:3, :3] = rot_interp.as_matrix()
        pose_interp[:3, 3] = t_interp
        interp_poses.append(pose_interp)
    
    return interp_poses

# ============================================================================
# Generate intermediate poses
# ============================================================================
all_poses = []
for i in range(len(poses) - 1):
    interp = interpolate_poses_slerp(poses[i], poses[i+1], args.num_interp)
    all_poses.extend(interp)

all_poses = np.array(all_poses)
if args.max_renders > 0:
    all_poses = all_poses[:args.max_renders]

print(f"  Generated {len(all_poses)} intermediate poses")
np.save(f"{PART3_OUT}/intermediate_poses.npy", all_poses)

# ============================================================================
# Get image size from real images
# ============================================================================
img_dir = f"{PART2_BASE}/images"
if os.path.exists(img_dir):
    img_files = list(Path(img_dir).glob("*.png"))
    if img_files:
        img = Image.open(img_files[0])
        W, H = img.size
        print(f"  Image size: {W}x{H}")
else:
    W, H = 1920, 1080
    print(f"  Using default image size: {W}x{H}")

# ============================================================================
# Create temporary scene for rendering
# ============================================================================
temp_scene = f"{PART3_OUT}/temp_scene"
os.makedirs(f"{temp_scene}/images", exist_ok=True)
os.makedirs(f"{temp_scene}/sparse/0", exist_ok=True)

# Create dummy images
print(f"  Creating dummy images...")
for i in range(len(all_poses)):
    dummy_img = Image.new('RGB', (W, H), color=(128, 128, 128))
    dummy_img.save(f"{temp_scene}/images/dummy_{i:04d}.png")

# Create cameras.bin
fx = fy = max(W, H) * 0.8
cx, cy = W / 2, H / 2

cam_path = f"{temp_scene}/sparse/0/cameras.bin"
with open(cam_path, "wb") as f:
    f.write(struct.pack("<Q", 1))  # num_cameras
    f.write(struct.pack("<I", 1))  # camera_id
    f.write(struct.pack("<i", 1))  # model_id (PINHOLE)
    f.write(struct.pack("<Q", W))  # width
    f.write(struct.pack("<Q", H))  # height
    for v in [fx, fy, cx, cy]:
        f.write(struct.pack("<d", float(v)))

# Create images.bin (minimal)
from scipy.spatial.transform import Rotation as R_quat

def rotmat_to_quat_bin(matrix):
    r = R_quat.from_matrix(matrix)
    q = r.as_quat()  # (x, y, z, w)
    return (q[3], q[0], q[1], q[2])  # (w, x, y, z)

img_path = f"{temp_scene}/sparse/0/images.bin"
with open(img_path, "wb") as f:
    f.write(struct.pack("<Q", len(all_poses)))
    for i, pose in enumerate(all_poses):
        w2c = np.linalg.inv(pose)
        qvec = rotmat_to_quat_bin(w2c[:3, :3])
        tvec = w2c[:3, 3]
        f.write(struct.pack("<I", i + 1))
        for v in qvec:
            f.write(struct.pack("<d", v))
        for v in tvec:
            f.write(struct.pack("<d", v))
        f.write(struct.pack("<I", 1))
        f.write(f"dummy_{i:04d}.png".encode() + b"\x00")
        f.write(struct.pack("<Q", 0))

# Create empty points3D.bin
pts_path = f"{temp_scene}/sparse/0/points3D.bin"
with open(pts_path, "wb") as f:
    f.write(struct.pack("<Q", 0))

print(f"  Created temp scene at {temp_scene}")

# ============================================================================
# Render using 3DGS render.py
# ============================================================================
print(f"\n  Rendering with 3DGS...")
render_cmd = f"python {GS_REPO}/render.py --model_path {MODEL_PATH} --source_path {temp_scene} --quiet"

import subprocess
import shutil
result = subprocess.run(render_cmd, shell=True, capture_output=True, text=True)

if result.returncode != 0:
    print(f"  Render failed: {result.stderr[:500]}")
else:
    # Copy renders from both test and train directories
    total_copied = 0
    for subdir in ["test", "train"]:
        render_source = Path(MODEL_PATH) / subdir / f"ours_{args.iterations}" / "renders"
        if render_source.exists():
            for f in render_source.glob("*.png"):
                shutil.copy(f, f"{PART3_OUT}/intermediate_renders/")
            num = len(list(render_source.glob("*.png")))
            total_copied += num
            print(f"  Copied {num} renders from {subdir}")
    
    print(f"  Total: {total_copied} renders saved to {PART3_OUT}/intermediate_renders/")

print("\n" + "=" * 70)
print("  Step 1 Complete!")
print(f"  Intermediate poses: {PART3_OUT}/intermediate_poses.npy")
print(f"  Rendered images: {PART3_OUT}/intermediate_renders/")
print("=" * 70)