import os
import numpy as np

# Import the helpers from your existing script
from visual_3D import (
    load_depth_npy,
    load_orientations_npy,
    bbox_centers_to_3d,
    ransac_line_3d,
    estimate_queue_forward_direction,
    classify_queue_facing_direction,
)

from eval_gpt import load_all_json_recursive_with_paths  # from your attached script

def load_image_paths(list_path: str):
    paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p and not p.startswith("#"):
                paths.append(p)
    return paths

IMAGE_LIST_PATH = "/scratch/ds5725/linefinder/LineFinder/code/olivia_luna_image_paths.txt"
# Find matching images
IMAGE_PATHS = load_image_paths(IMAGE_LIST_PATH)

print(f"Found {len(IMAGE_PATHS)} images that have GT JSONs.")
# for p in matched_images:
#     print(p)

# Optional: save to file
# out_file = "images_with_gt.txt"
# with open(out_file, "w") as f:
#     for p in matched_images:
#         f.write(p + "\n")

# print(f"\nSaved list to {out_file}")
# exit()
# -------------------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------------------

# List of image paths to process (edit this list)
# IMAGE_PATHS = [
#     r"D:\cv\LineFinder\queue_images\IMG_5954.JPG",
#     r"D:\cv\LineFinder\queue_images\IMG_5955.JPG",
#     r"D:\cv\LineFinder\queue_images\IMG_5956.JPG",
# ]


def load_focal_lengths(txt_path):
    """
    Reads focal_length_px.txt and returns:
        { 'IMG_5954': 441.63, 'IMG_5955': 440.83, ... }
    """
    focal_dict = {}
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            name, val = line.strip().split()
            focal_dict[name] = float(val)
    return focal_dict

# Where the depth .npy and bbox/orient .npy files live
DEPTH_DIR = "/scratch/ds5725/linefinder/LineFinder/depth_map"
BBOX_ORIENT_DIR = "/scratch/ds5725/linefinder/LineFinder/bbox_orient"

# Focal length in pixels (set to whatever you used in 3D_visual.py)
FOCAL_TXT = "/scratch/ds5725/linefinder/LineFinder/focal_length_px.txt"
focal_dict = load_focal_lengths(FOCAL_TXT)

print(f"Loaded {len(focal_dict)} focal-length records")

# RANSAC settings (same as your single-image script)
RANSAC_NUM_ITERS = 1000
RANSAC_DIST_THRESH = 0.8
RANSAC_MIN_INLIER_RATIO = 0.3

# front/back tolerance in degrees (same logic as before)
FRONT_BACK_TOL_DEG = 45.0


# -------------------------------------------------------------------
# HELPER: process one image
# -------------------------------------------------------------------

def classify_queue_for_image(
    image_path: str,
    depth_dir: str,
    bbox_orient_dir: str,
    f_px: float,
) -> str:
    """
    Given an RGB image path, load the corresponding depth, bbox and
    orientation npy files, run the 3D line + orientation logic, and
    return the facing direction label (away / towards / sideways-* / unknown).
    """
    if not os.path.isfile(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return "missing-image"

    # Get image ID from filename, e.g. IMG_5955 from IMG_5955.JPG
    base = os.path.basename(image_path)
    image_id, _ = os.path.splitext(base)  # ("IMG_5955", ".JPG")

    depth_npy_path = os.path.join(depth_dir, image_id + ".npy")
    bboxes_npy_path = os.path.join(bbox_orient_dir, image_id + "_bboxes.npy")
    orient_npy_path = os.path.join(bbox_orient_dir, image_id + "_orient.npy")

    # Check all required files
    missing_any = False
    for p in [depth_npy_path, bboxes_npy_path, orient_npy_path]:
        if not os.path.isfile(p):
            print(f"[WARN] Missing file for {image_id}: {p}")
            missing_any = True
    if missing_any:
        return "missing-data"

    # ----- Load data -----
    depth = load_depth_npy(depth_npy_path)
    bboxes = np.load(bboxes_npy_path).astype(np.float32)
    orientations_deg = load_orientations_npy(orient_npy_path)

    # ----- 2D -> 3D -----
    points_3d, centers_uv = bbox_centers_to_3d(bboxes, depth, f_px)

    if points_3d.shape[0] < 2:
        print(f"[INFO] Not enough 3D points to fit a line for {image_id}.")
        return "too-few-points"

    # ----- RANSAC line fit -----
    line_point, line_dir, inlier_mask = ransac_line_3d(
        points_3d,
        num_iters=RANSAC_NUM_ITERS,
        dist_thresh=RANSAC_DIST_THRESH,
        min_inliers_ratio=RANSAC_MIN_INLIER_RATIO,
    )

    # ----- Use orientations to pick line direction -----
    queue_forward_dir_3d, score = estimate_queue_forward_direction(
        line_dir_3d=line_dir,
        orientations_deg=orientations_deg,
        inlier_mask=inlier_mask,
    )
    print(f"[DEBUG] {image_id}: avg alignment score = {score:.3f}")

    # ----- Classify facing direction -----
    facing_label = classify_queue_facing_direction(
        queue_forward_dir_3d,
        front_back_tolerance_deg=FRONT_BACK_TOL_DEG,
    )

    return facing_label


# -------------------------------------------------------------------
# MAIN: loop over all image paths
# -------------------------------------------------------------------
if __name__ == "__main__":

    results = {}

    for img_path in IMAGE_PATHS:
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        # lookup focal length
        if image_id not in focal_dict:
            print(f"[WARN] No focal length found for {image_id}, skipping")
            continue
        label = classify_queue_for_image(
            image_path=img_path,
            depth_dir=DEPTH_DIR,
            bbox_orient_dir=BBOX_ORIENT_DIR,
            f_px=focal_dict[image_id],
        )
        results[img_path] = label
        print(f"{img_path} -> {label}")

    # (Optional) write to a CSV file
    out_csv = "OL_queue_facing_results.csv"
    try:
        import csv

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "facing_direction"])
            for img_path, label in results.items():
                writer.writerow([img_path, label])
        print(f"\nSaved results to {out_csv}")
    except Exception as e:
        print(f"[WARN] Could not save CSV: {e}")
