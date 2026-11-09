import os
import csv
import numpy as np

# Import the helpers from your existing script
from visual_3D import (
    load_depth_npy,
    bbox_centers_to_3d,
    ransac_line_3d,
)

def load_image_paths(list_path: str):
    paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p and not p.startswith("#"):
                paths.append(p)
    return paths


def load_focal_lengths(txt_path):
    """
    Reads focal_length_px.txt and returns:
        { 'IMG_5954': 441.63, 'IMG_5955': 440.83, ... }
    """
    focal_dict = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            name, val = line.strip().split()
            focal_dict[name] = float(val)
    return focal_dict


# -------------------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------------------

IMAGE_LIST_PATH = "/mnt/NAS/home/ds5725/linefinder/code/old_image_paths.txt"

DEPTH_DIR = "/mnt/NAS/home/ds5725/linefinder/depth_map"
BBOX_ORIENT_DIR = "/mnt/NAS/home/ds5725/linefinder/bbox_orient"
FOCAL_TXT = "/mnt/NAS/home/ds5725/linefinder/focal_length_px.txt"

OUT_CSV = "OLD_number_of_people.csv"

# RANSAC settings
RANSAC_NUM_ITERS = 1000
RANSAC_DIST_THRESH = 0.8
RANSAC_MIN_INLIER_RATIO = 0.3


# -------------------------------------------------------------------
# HELPER: process one image
# -------------------------------------------------------------------

def count_people_for_image(
    image_path: str,
    depth_dir: str,
    bbox_orient_dir: str,
    f_px: float,
):
    """
    Returns:
        number_of_people = inlier_mask.sum() after RANSAC
    """
    if not os.path.isfile(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return None, "missing-image"

    base = os.path.basename(image_path)
    image_id, _ = os.path.splitext(base)

    depth_npy_path = os.path.join(depth_dir, image_id + ".npy")
    bboxes_npy_path = os.path.join(bbox_orient_dir, image_id + "_bboxes.npy")

    missing_any = False
    for p in [depth_npy_path, bboxes_npy_path]:
        if not os.path.isfile(p):
            print(f"[WARN] Missing file for {image_id}: {p}")
            missing_any = True
    if missing_any:
        return None, "missing-data"

    # ----- Load data -----
    depth = load_depth_npy(depth_npy_path)
    bboxes = np.load(bboxes_npy_path).astype(np.float32)

    if bboxes.shape[0] < 2:
        print(f"[INFO] Not enough bboxes for {image_id}.")
        return None, "too-few-bboxes"

    # ----- 2D -> 3D -----
    points_3d, centers_uv = bbox_centers_to_3d(bboxes, depth, f_px)

    if points_3d.shape[0] < 2:
        print(f"[INFO] Not enough valid 3D points for {image_id}.")
        return None, "too-few-points"

    # ----- RANSAC line fit -----
    line_point, line_dir, inlier_mask = ransac_line_3d(
        points_3d,
        num_iters=RANSAC_NUM_ITERS,
        dist_thresh=RANSAC_DIST_THRESH,
        min_inliers_ratio=RANSAC_MIN_INLIER_RATIO,
    )

    if inlier_mask is None:
        print(f"[WARN] RANSAC failed for {image_id}")
        return None, "ransac-failed"

    number_of_people = int(inlier_mask.sum())
    return number_of_people, "ok"


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    IMAGE_PATHS = load_image_paths(IMAGE_LIST_PATH)
    focal_dict = load_focal_lengths(FOCAL_TXT)

    print(f"Found {len(IMAGE_PATHS)} images to process.")
    print(f"Loaded {len(focal_dict)} focal-length records.")

    rows = []

    for img_path in IMAGE_PATHS:
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        if image_id not in focal_dict:
            print(f"[WARN] No focal length found for {image_id}, skipping")
            rows.append([img_path, image_id, "", "missing-focal"])
            continue

        count, status = count_people_for_image(
            image_path=img_path,
            depth_dir=DEPTH_DIR,
            bbox_orient_dir=BBOX_ORIENT_DIR,
            f_px=focal_dict[image_id],
        )

        if count is None:
            print(f"{img_path} -> {status}")
            rows.append([img_path, image_id, "", status])
        else:
            print(f"{img_path} -> {count}")
            rows.append([img_path, image_id, count, status])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "image_id", "OL_number_of_people", "status"])
        writer.writerows(rows)

    print(f"\nSaved results to {OUT_CSV}")