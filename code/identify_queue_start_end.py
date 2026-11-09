import os
import argparse
import numpy as np
import cv2

from visual_3D import (
    load_depth_npy,
    load_orientations_npy,
    bbox_centers_to_3d,
    project_points_3d_to_2d,
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

# def pick_start_end_indices(points_3d,
#                            line_point,
#                            queue_forward_dir_3d,
#                            inlier_mask=None,
#                            gap_scale=3.0,
#                            max_gap=None):
#     """
#     Robust endpoint selection:
#     - Project points onto queue direction -> 1D coordinate t
#     - Sort by t
#     - Split into contiguous segments using neighbor gaps
#     - Take endpoints from the largest segment (the actual queue)
    
#     START = head (max t), END = tail (min t)

#     gap_scale controls how strict contiguity is:
#       max_gap = gap_scale * median_neighbor_gap (if max_gap is None)
#     """
#     q = np.asarray(queue_forward_dir_3d, dtype=np.float32)
#     q = q / (np.linalg.norm(q) + 1e-8)

#     t_vals = (points_3d - line_point[None, :]) @ q  # (N,)

#     # restrict candidate points
#     if inlier_mask is not None and inlier_mask.shape[0] == points_3d.shape[0]:
#         idx_pool = np.flatnonzero(inlier_mask)
#     else:
#         idx_pool = np.arange(points_3d.shape[0])

#     if idx_pool.size < 2:
#         i = int(idx_pool[0]) if idx_pool.size == 1 else 0
#         return i, i, t_vals

#     # sort pool by t
#     pool_t = t_vals[idx_pool]
#     order = np.argsort(pool_t)
#     idx_sorted = idx_pool[order]
#     t_sorted = pool_t[order]

#     gaps = np.diff(t_sorted)
#     if gaps.size == 0:
#         i = int(idx_sorted[0])
#         return i, i, t_vals

#     # auto threshold based on typical spacing along the line
#     if max_gap is None:
#         med_gap = float(np.median(gaps))
#         med_gap = max(med_gap, 1e-3)
#         max_gap = gap_scale * med_gap

#     # find contiguous segments where gap <= max_gap
#     breaks = np.where(gaps > max_gap)[0]
#     seg_starts = np.r_[0, breaks + 1]
#     seg_ends = np.r_[breaks, len(t_sorted) - 1]
#     seg_lengths = seg_ends - seg_starts + 1

#     # pick the largest segment = main queue chain
#     best = int(np.argmax(seg_lengths))
#     s0, s1 = int(seg_starts[best]), int(seg_ends[best])

#     chain_idx = idx_sorted[s0:s1 + 1]
#     chain_t = t_sorted[s0:s1 + 1]

#     # endpoints within the chain
#     start_idx = int(chain_idx[np.argmax(chain_t)])  # head
#     end_idx   = int(chain_idx[np.argmin(chain_t)])  # tail
#     return start_idx, end_idx, t_vals

def pick_start_end_indices(points_3d,
                           centers_uv,
                           depth_shape,
                           f_px,
                           line_point,
                           queue_forward_dir_3d,
                           inlier_mask=None,
                           axis_extent_3d=1.0):
    """
    Pick queue endpoints in image space.

    Steps:
    - Use the fitted 3D queue direction to build a short 3D axis around line_point.
    - Project that 3D axis back to 2D using the same camera model.
    - Project each candidate bbox center in 2D onto this 2D axis to get a 1D coordinate s.
    - START = point with maximum s (head of queue).
    - END   = point with minimum s (tail of queue).

    This keeps the queue direction estimate in 3D, but makes the endpoint ordering
    happen in the image plane after projection.
    """
    q = np.asarray(queue_forward_dir_3d, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    if inlier_mask is not None and inlier_mask.shape[0] == points_3d.shape[0]:
        idx_pool = np.flatnonzero(inlier_mask)
    else:
        idx_pool = np.arange(points_3d.shape[0])

    if idx_pool.size == 0:
        return 0, 0, np.zeros((points_3d.shape[0],), dtype=np.float32), None, None
    if idx_pool.size == 1:
        i = int(idx_pool[0])
        s_vals = np.full((points_3d.shape[0],), np.nan, dtype=np.float32)
        s_vals[i] = 0.0
        return i, i, s_vals, None, None

    # Build a short 3D axis centered on the fitted line point and aligned with the queue direction.
    axis_pts_3d = np.stack([
        line_point - axis_extent_3d * q,
        line_point + axis_extent_3d * q,
    ], axis=0)
    axis_uv = project_points_3d_to_2d(axis_pts_3d, depth_shape, f_px)

    if axis_uv.shape[0] != 2:
        # Fallback: if projection is degenerate, order directly in 3D.
        t_vals = (points_3d - line_point[None, :]) @ q
        pool_t = t_vals[idx_pool]
        start_idx = int(idx_pool[np.argmax(pool_t)])
        end_idx = int(idx_pool[np.argmin(pool_t)])
        return start_idx, end_idx, t_vals.astype(np.float32), None, None

    axis_dir_2d = axis_uv[1] - axis_uv[0]
    axis_norm = float(np.linalg.norm(axis_dir_2d))
    if axis_norm < 1e-6:
        t_vals = (points_3d - line_point[None, :]) @ q
        pool_t = t_vals[idx_pool]
        start_idx = int(idx_pool[np.argmax(pool_t)])
        end_idx = int(idx_pool[np.argmin(pool_t)])
        return start_idx, end_idx, t_vals.astype(np.float32), axis_uv[0], axis_uv[1]

    axis_dir_2d = axis_dir_2d / axis_norm

    s_vals = np.full((points_3d.shape[0],), np.nan, dtype=np.float32)
    rel_uv = centers_uv - axis_uv[0][None, :]
    s_vals[:] = rel_uv @ axis_dir_2d

    pool_s = s_vals[idx_pool]
    start_idx = int(idx_pool[np.argmax(pool_s)])
    end_idx = int(idx_pool[np.argmin(pool_s)])
    return start_idx, end_idx, s_vals, axis_uv[0], axis_uv[1]

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
    orientations_deg_all = load_orientations_npy(orient_npy_path)  # (N,)

    # IMPORTANT: bbox_centers_to_3d filters out invalid depth centers, so we must build a mapping
    points_3d, centers_uv = bbox_centers_to_3d(bboxes_all, depth, f_px)

    # Rebuild the valid->orig mapping by re-running the same validity test (mirrors bbox_centers_to_3d)
    H, W = depth.shape
    fx = fy = float(f_px)
    cx = W / 2.0
    cy = H / 2.0

    valid_to_orig_idx = []
    valid_orientations = []
    valid_bboxes = []
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
        valid_bboxes.append(bboxes_all[i])

    valid_to_orig_idx = np.array(valid_to_orig_idx, dtype=np.int64)
    valid_orientations = np.array(valid_orientations, dtype=np.float32)
    valid_bboxes = np.array(valid_bboxes, dtype=np.float32)

    # orig_N = bboxes_all.shape[0]
    # valid_set = set(valid_to_orig_idx.tolist())
    # dropped = [i for i in range(orig_N) if i not in valid_set]
    # print(f"Total bboxes: {orig_N}, valid(3D): {len(valid_to_orig_idx)}, dropped: {len(dropped)}")
    # print("Dropped indices (first 20):", dropped[:20])

    if points_3d.shape[0] < 2:
        raise ValueError("Not enough valid 3D points to fit a queue line.")

    # Fit line with RANSAC
    line_point, line_dir, inlier_mask = ransac_line_3d(
        points_3d,
        num_iters=ransac_num_iters,
        dist_thresh=ransac_dist_thresh,
        min_inliers_ratio=ransac_min_inliers_ratio,
    )

    # Choose sign of direction using orientations (this is your queue direction)
    queue_forward_dir_3d, score = estimate_queue_forward_direction(
        line_dir_3d=line_dir,
        orientations_deg=valid_orientations,
        inlier_mask=inlier_mask,
    )

    # Pick endpoints along the signed queue direction
    start_valid_idx, end_valid_idx, s_vals, axis_uv_start, axis_uv_end = pick_start_end_indices(
        points_3d=points_3d,
        centers_uv=centers_uv,
        depth_shape=depth.shape,
        f_px=f_px,
        line_point=line_point,
        queue_forward_dir_3d=queue_forward_dir_3d,
        inlier_mask=inlier_mask,
    )

    start_orig_idx = int(valid_to_orig_idx[start_valid_idx])
    end_orig_idx   = int(valid_to_orig_idx[end_valid_idx])

    return {
        "score": float(score),
        "queue_forward_dir_3d": queue_forward_dir_3d,
        "line_point": line_point,
        "line_dir": line_dir,
        "inlier_mask": inlier_mask,
        "s_vals": s_vals,
        "axis_uv_start": axis_uv_start,
        "axis_uv_end": axis_uv_end,

        "start_valid_idx": start_valid_idx,
        "end_valid_idx": end_valid_idx,
        "start_orig_idx": start_orig_idx,
        "end_orig_idx": end_orig_idx,

        "start_bbox_xyxy": bboxes_all[start_orig_idx],
        "end_bbox_xyxy": bboxes_all[end_orig_idx],
    }

# -----------------------------
# Optional: visualization
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
        cv2.putText(im, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)

    # Colors are BGR
    draw_box(img, start_bbox, "START (head)", (0, 255, 255))  # yellow
    draw_box(img, end_bbox,   "END (tail)",   (0, 255, 0))    # green

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"Saved visualization: {out_path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_id", required=True)
    ap.add_argument("--root", default="/mnt/NAS/home/ds5725/linefinder")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    image_id = args.image_id
    root = args.root

    image_path = os.path.join(root, "Images/QueuesOutdoors", f"{image_id}.jpg")
    depth_path = os.path.join(root, "depth_map", f"{image_id}.npy")
    bbox_path  = os.path.join(root, "bbox_orient", f"{image_id}_bboxes.npy")
    orient_path = os.path.join(root, "bbox_orient", f"{image_id}_orient.npy")
    fpx_path = os.path.join(root, "focal_length_px.txt")

    f_px = load_fpx_from_txt(fpx_path, image_id)

    if args.out is None:
        args.out = f"{image_id}_start_end.jpg"

    res = identify_start_end_bboxes(
        image_path=image_path,
        depth_npy_path=depth_path,
        bboxes_npy_path=bbox_path,
        orient_npy_path=orient_path,
        f_px=f_px,
    )


    print("avg alignment score:", res["score"])
    print("start_orig_idx:", res["start_orig_idx"], "start_bbox:", res["start_bbox_xyxy"].tolist())
    print("end_orig_idx:",   res["end_orig_idx"],   "end_bbox:",   res["end_bbox_xyxy"].tolist())

    visualize_start_end_on_image(
        image_path=image_path,
        start_bbox=res["start_bbox_xyxy"],
        end_bbox=res["end_bbox_xyxy"],
        out_path=args.out,
    )

if __name__ == "__main__":
    main()
