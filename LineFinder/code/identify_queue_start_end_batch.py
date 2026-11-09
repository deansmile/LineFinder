import os
import argparse
import numpy as np
import cv2

from visual_3D import (
    load_depth_npy,
    load_orientations_npy,
    bbox_centers_to_3d,
    ransac_line_3d,
    estimate_queue_forward_direction,
)

# -----------------------------
# Core: pick start/end boxes
# -----------------------------
def load_fpx_from_txt(txt_path: str, image_id: str) -> float:
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            k, v = line.strip().split()
            if k == image_id:
                return float(v)
    raise KeyError(f"fpx not found for image_id={image_id} in {txt_path}")


def pick_start_end_indices(points_3d,
                           line_point,
                           queue_forward_dir_3d,
                           inlier_mask=None,
                           gap_scale=3.0,
                           max_gap=None):
    """
    Robust endpoint selection:
    - Project points onto queue direction -> 1D coordinate t
    - Sort by t
    - Split into contiguous segments using neighbor gaps
    - Take endpoints from the largest segment (the actual queue)

    START = head (max t), END = tail (min t)

    gap_scale controls how strict contiguity is:
      max_gap = gap_scale * median_neighbor_gap (if max_gap is None)
    """
    q = np.asarray(queue_forward_dir_3d, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    t_vals = (points_3d - line_point[None, :]) @ q  # (N,)

    if inlier_mask is not None and inlier_mask.shape[0] == points_3d.shape[0]:
        idx_pool = np.flatnonzero(inlier_mask)
    else:
        idx_pool = np.arange(points_3d.shape[0])

    if idx_pool.size < 2:
        i = int(idx_pool[0]) if idx_pool.size == 1 else 0
        return i, i, t_vals

    pool_t = t_vals[idx_pool]
    order = np.argsort(pool_t)
    idx_sorted = idx_pool[order]
    t_sorted = pool_t[order]

    gaps = np.diff(t_sorted)
    if gaps.size == 0:
        i = int(idx_sorted[0])
        return i, i, t_vals

    if max_gap is None:
        med_gap = float(np.median(gaps))
        med_gap = max(med_gap, 1e-3)
        max_gap = gap_scale * med_gap

    breaks = np.where(gaps > max_gap)[0]
    seg_starts = np.r_[0, breaks + 1]
    seg_ends = np.r_[breaks, len(t_sorted) - 1]
    seg_lengths = seg_ends - seg_starts + 1

    best = int(np.argmax(seg_lengths))
    s0, s1 = int(seg_starts[best]), int(seg_ends[best])

    chain_idx = idx_sorted[s0:s1 + 1]
    chain_t = t_sorted[s0:s1 + 1]

    start_idx = int(chain_idx[np.argmax(chain_t)])  # head
    end_idx   = int(chain_idx[np.argmin(chain_t)])  # tail
    return start_idx, end_idx, t_vals


def identify_start_end_bboxes(
    image_path: str,
    depth_npy_path: str,
    bboxes_npy_path: str,
    orient_npy_path: str,
    f_px: float,
    ransac_num_iters: int = 1000,
    ransac_dist_thresh: float = 0.8,
    ransac_min_inliers_ratio: float = 0.3
):
    """
    Returns a dict with:
      - start_bbox_xyxy, end_bbox_xyxy   (in original bbox file indexing)
      - start_valid_idx, end_valid_idx   (indices into the valid-depth subset)
      - valid_to_orig_idx               (mapping from valid-depth idx -> original bbox idx)
      - queue_forward_dir_3d, line_point, line_dir, inlier_mask
    """
    depth = load_depth_npy(depth_npy_path)
    bboxes_all = np.load(bboxes_npy_path).astype(np.float32)
    orientations_deg_all = load_orientations_npy(orient_npy_path)

    points_3d, centers_uv = bbox_centers_to_3d(bboxes_all, depth, f_px)

    H, W = depth.shape

    valid_to_orig_idx = []
    valid_orientations = []

    for i, (x1, y1, x2, y2) in enumerate(bboxes_all):
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))
        u = int(np.clip(u, 0, W - 1))
        v = int(np.clip(v, 0, H - 1))
        Z = float(depth[v, u])
        if not np.isfinite(Z) or Z <= 0:
            continue
        valid_to_orig_idx.append(i)
        valid_orientations.append(orientations_deg_all[i])

    valid_to_orig_idx = np.array(valid_to_orig_idx, dtype=np.int64)
    valid_orientations = np.array(valid_orientations, dtype=np.float32)

    if points_3d.shape[0] < 2:
        raise ValueError("Not enough valid 3D points to fit a queue line.")

    line_point, line_dir, inlier_mask = ransac_line_3d(
        points_3d,
        num_iters=ransac_num_iters,
        dist_thresh=ransac_dist_thresh,
        min_inliers_ratio=ransac_min_inliers_ratio,
    )

    queue_forward_dir_3d, score = estimate_queue_forward_direction(
        line_dir_3d=line_dir,
        orientations_deg=valid_orientations,
        inlier_mask=inlier_mask,
    )

    start_valid_idx, end_valid_idx, t_vals = pick_start_end_indices(
        points_3d=points_3d,
        line_point=line_point,
        queue_forward_dir_3d=queue_forward_dir_3d,
        inlier_mask=inlier_mask
    )

    start_orig_idx = int(valid_to_orig_idx[start_valid_idx])
    end_orig_idx   = int(valid_to_orig_idx[end_valid_idx])

    return {
        "score": float(score),
        "queue_forward_dir_3d": queue_forward_dir_3d,
        "line_point": line_point,
        "line_dir": line_dir,
        "inlier_mask": inlier_mask,
        "t_vals": t_vals,
        "start_valid_idx": start_valid_idx,
        "end_valid_idx": end_valid_idx,
        "start_orig_idx": start_orig_idx,
        "end_orig_idx": end_orig_idx,
        "start_bbox_xyxy": bboxes_all[start_orig_idx],
        "end_bbox_xyxy": bboxes_all[end_orig_idx],
    }


# -----------------------------
# Visualization
# -----------------------------
def visualize_start_end_on_image(image_path: str,
                                 start_bbox: np.ndarray,
                                 end_bbox: np.ndarray,
                                 out_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    def draw_box(im, box, label, color):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)
        cv2.putText(
            im, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA
        )

    draw_box(img, start_bbox, "START (head)", (0, 255, 255))  # yellow
    draw_box(img, end_bbox,   "END (tail)",   (0, 255, 0))    # green

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"Saved visualization: {out_path}")


# -----------------------------
# Batch helpers
# -----------------------------
def collect_all_images(image_dirs):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = []
    for folder in image_dirs:
        if not os.path.isdir(folder):
            print(f"Warning: folder does not exist, skipping: {folder}")
            continue
        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1]
            if ext in exts:
                image_paths.append(os.path.join(folder, name))
    return sorted(image_paths)


def process_one_image(image_path, root, vis_dir=None):
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    depth_path = os.path.join(root, "depth_map", f"{image_id}.npy")
    bbox_path = os.path.join(root, "bbox_orient", f"{image_id}_bboxes.npy")
    orient_path = os.path.join(root, "bbox_orient", f"{image_id}_orient.npy")
    fpx_path = os.path.join(root, "focal_length_px.txt")

    missing = []
    for p in [depth_path, bbox_path, orient_path, fpx_path]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            f"Missing required file(s) for {image_id}:\n" + "\n".join(missing)
        )

    f_px = load_fpx_from_txt(fpx_path, image_id)

    res = identify_start_end_bboxes(
        image_path=image_path,
        depth_npy_path=depth_path,
        bboxes_npy_path=bbox_path,
        orient_npy_path=orient_path,
        f_px=f_px,
    )

    print("=" * 80)
    print(f"image_id: {image_id}")
    print(f"image_path: {image_path}")
    print("avg alignment score:", res["score"])
    print("start_orig_idx:", res["start_orig_idx"], "start_bbox:", res["start_bbox_xyxy"].tolist())
    print("end_orig_idx:",   res["end_orig_idx"],   "end_bbox:",   res["end_bbox_xyxy"].tolist())

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
        out_path = os.path.join(vis_dir, f"{image_id}_start_end.jpg")
        visualize_start_end_on_image(
            image_path=image_path,
            start_bbox=res["start_bbox_xyxy"],
            end_bbox=res["end_bbox_xyxy"],
            out_path=out_path,
        )

    return {
        "image_id": image_id,
        "image_path": image_path,
        "score": res["score"],
        "start_orig_idx": res["start_orig_idx"],
        "end_orig_idx": res["end_orig_idx"],
        "start_bbox_xyxy": res["start_bbox_xyxy"].tolist(),
        "end_bbox_xyxy": res["end_bbox_xyxy"].tolist(),
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/ds5725/linefinder/LineFinder")
    ap.add_argument(
        "--vis_dir",
        default="/scratch/ds5725/linefinder/LineFinder/start_end_vis",
        help="Directory to save visualization images. Set to empty string to disable."
    )
    args = ap.parse_args()

    root = args.root

    image_dirs = [
        os.path.join(root, "Images", "QueuesInSupermarketNew"),
        os.path.join(root, "Images", "QueuesInThemeParks"),
        os.path.join(root, "Images", "QueuesOutdoors"),
    ]

    vis_dir = args.vis_dir if args.vis_dir != "" else None

    image_paths = collect_all_images(image_dirs)
    print(f"Found {len(image_paths)} images total.")

    success = 0
    failed = 0
    failed_images = []

    for image_path in image_paths:
        try:
            process_one_image(image_path, root=root, vis_dir=vis_dir)
            success += 1
        except Exception as e:
            failed += 1
            failed_images.append((image_path, str(e)))
            print("=" * 80)
            print(f"Failed: {image_path}")
            print(e)

    print("\n" + "#" * 80)
    print("Done.")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")

    if failed_images:
        print("\nFailed image list:")
        for image_path, err in failed_images:
            print(f"{image_path} -> {err}")


if __name__ == "__main__":
    main()