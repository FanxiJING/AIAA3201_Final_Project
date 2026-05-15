# -*- coding: utf-8 -*-

"""
Part 3: Pseudo-View Generation with Difix3D
Use nearest real frame as reference for each intermediate render
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path

sys.path.insert(0, '/data2/fjing221/workplace/Difix3D/src')

import torch
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
from PIL import Image

# ============================================
# Configuration
# ============================================

DATASETS = ["dl3dv", "re10k", "waymo"]
METHODS = ["instantsplat", "reggs"]

BASE_PATH = "/data2/fjing221/workplace/output/part3_intermediate"
REAL_IMAGES_PATH = "/data2/fjing221/workplace/output/part2_alternative"
OUTPUT_BASE = "/data2/fjing221/workplace/output/part3_enhanced"

# ============================================
# Load Difix3D Model
# ============================================

print("Loading Difix3D model...")

LOCAL_MODEL_PATH = "/data2/fjing221/difix_ref_model"

pipe = DifixPipeline.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

device = torch.device("cuda:0")
pipe.to(device)
print("Model loaded!")

# ============================================
# Helper: find nearest real frame for each render
# ============================================

import cv2

def compute_confidence_score(pseudo_path, real_path_A, real_path_B):
    """
    Compute a single confidence score (0-1) for a pseudo-view.
    Higher score means more consistent with real frames.
    """
    # Load images as grayscale
    img_pseudo = cv2.imread(pseudo_path, 0)
    img_real_A = cv2.imread(real_path_A, 0)
    
    if img_pseudo is None or img_real_A is None:
        return 0.5  # default fallback
    
    # SIFT feature matching
    sift = cv2.SIFT_create()
    kp_p, des_p = sift.detectAndCompute(img_pseudo, None)
    kp_A, des_A = sift.detectAndCompute(img_real_A, None)
    
    if des_p is None or des_A is None:
        return 0.5
    
    # Match with first real frame
    bf = cv2.BFMatcher()
    matches_A = bf.knnMatch(des_p, des_A, k=2) if des_A is not None else []
    
    # Lowe's ratio test
    good_A = [m for m, n in matches_A if m.distance < 0.75 * n.distance] if matches_A else []
    
    # Confidence = normalized match count
    max_matches = max(len(kp_p), 1)
    confidence = len(good_A) / max_matches
    confidence = min(1.0, confidence)
    
    return confidence

def find_nearest_real_frame(render_idx, num_real_frames, renders_per_interval):
    if renders_per_interval <= 0:
        return 0
    
    interval = render_idx // renders_per_interval
    real_idx = min(interval, num_real_frames - 2)
    real_idx = max(real_idx, 0)
    
    return real_idx

def get_real_image_path(dataset, method, real_idx):
    """Get path to real image from Part 2 output"""
    img_dir = f"{REAL_IMAGES_PATH}/{dataset}/{method}/images"
    
    if os.path.exists(img_dir):
        images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if real_idx < len(images):
            return images[real_idx]
    
    return None

# ============================================
# Main Process
# ============================================

def main():
    for dataset in DATASETS:
        for method in METHODS:
            print(f"\n{'='*50}")
            print(f"Processing: {dataset} / {method}")
            print(f"{'='*50}")
            
            renders_dir = os.path.join(BASE_PATH, dataset, method, "intermediate_renders")
            
            if not os.path.exists(renders_dir):
                print(f"  SKIP: No renders found at {renders_dir}")
                continue
            
            render_files = sorted(glob.glob(os.path.join(renders_dir, "*.png")))
            print(f"  Found {len(render_files)} render images")
            
            if len(render_files) == 0:
                continue
            
            # Determine number of renders per interval
            poses_path = os.path.join(BASE_PATH, dataset, method, "intermediate_poses.npy")
            real_poses_path = f"{REAL_IMAGES_PATH}/{dataset}/{method}/poses_c2w.npy"
            
            num_renders = len(render_files)
            num_real = 11  # default
            
            if os.path.exists(real_poses_path):
                real_poses = np.load(real_poses_path)
                num_real = len(real_poses)
            
            if num_real > 1:
                renders_per_interval = num_renders // (num_real - 1)
                if renders_per_interval == 0:
                    renders_per_interval = 1
            else:
                renders_per_interval = 1
            
            print(f"  Real frames: {num_real}, Renders per interval: {renders_per_interval}")
            
            # Create output directory
            output_dir = os.path.join(OUTPUT_BASE, dataset, method, "pseudo_views")
            os.makedirs(output_dir, exist_ok=True)
            
            success_count = 0
            skip_count = 0
            fail_count = 0
            
            for i, render_path in enumerate(render_files):
                filename = os.path.basename(render_path)
                base_name = os.path.splitext(filename)[0]
                confidence_filename = f"pseudo_{base_name}_confidence.npy"
                output_path = os.path.join(output_dir, filename)
                confidence_path = os.path.join(output_dir, confidence_filename)
                
                if os.path.exists(output_path):
                    print(f"  [{i+1}/{len(render_files)}] Skipping {filename}")
                    skip_count += 1
                    continue
                
                # Find nearest real frame as reference
                real_idx = find_nearest_real_frame(i, num_real, renders_per_interval)
                real_img_path = get_real_image_path(dataset, method, real_idx)
                
                # Also get the other adjacent real frame for consistency check
                real_idx_B = min(real_idx + 1, num_real - 1)
                real_img_path_B = get_real_image_path(dataset, method, real_idx_B)
                
                if real_img_path is None or not os.path.exists(real_img_path):
                    print(f"  [{i+1}/{len(render_files)}] WARNING: No real image for index {real_idx}, using render as ref")
                    real_img_path = render_path
                    real_img_path_B = render_path
                
                print(f"  [{i+1}/{len(render_files)}] Processing {filename} (ref: real_{real_idx})...")
                
                try:
                    render_img = load_image(render_path)
                    ref_img = load_image(real_img_path)
                    
                    with torch.no_grad():
                        enhanced = pipe(
                            prompt="remove degradation",
                            image=render_img,
                            ref_image=ref_img,
                            num_inference_steps=1,
                            timesteps=[199],
                            guidance_scale=0.0
                        ).images[0]
                    
                    # Save enhanced image
                    enhanced.save(output_path)
                    
                    # ============================================
                    # Compute and save confidence score
                    # ============================================
                    confidence = compute_confidence_score(
                        render_path, 
                        real_img_path, 
                        real_img_path_B
                    )
                    np.save(confidence_path, np.array([confidence]))
                    print(f"    Saved: {output_path} (confidence: {confidence:.3f})")
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    fail_count += 1
            
            print(f"\n  Summary: Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")
    
    print("\n" + "="*50)
    print("All done!")
    print("="*50)

if __name__ == "__main__":
    main()