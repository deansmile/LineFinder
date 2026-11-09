import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import cv2

def load_depth_npy(depth_npy_path: str) -> np.ndarray:
    """Load metric depth map (in meters) saved as .npy from Depth-Pro."""
    depth = np.load(depth_npy_path)  # already in meters
    return depth.astype(np.float32)

def load_orientations_npy(orient_npy_path: str) -> np.ndarray:
    """Load body-orientation angles (degrees) from a .npy file."""
    ori = np.load(orient_npy_path)
    ori = np.asarray(ori, dtype=np.float32)
    # Handle shape (N,1) or (N,)
    if ori.ndim > 1:
        ori = np.squeeze(ori)
    return ori  # shape (N,)

def visualize_depth_map(depth_npy_path, f_px=None, save_path=None):
    """
    Visualize a Depth Pro depth map (saved as .npy or .npz) using the same
    inverse-depth normalization as the official run.py.

    Args:
        depth_npy_path: path to a .npy or .npz file containing depth.
                        If .npz, it should have key 'depth'.
        f_px: focal length in pixels (not needed for visualization, kept for API compatibility).
        save_path: optional path to save the color depth image (e.g. 'depth_viz.jpg').
    """
    data = np.load(depth_npy_path)

    # Support both .npy (array) and .npz (with key 'depth')
    if isinstance(data, np.lib.npyio.NpzFile):
        depth = data["depth"].astype(np.float32)
    else:
        depth = data.astype(np.float32)

    # Compute inverse depth (1 / depth)
    inverse_depth = 1.0 / depth

    # Handle non-finite values (just in case)
    valid_mask = np.isfinite(inverse_depth)
    if not np.any(valid_mask):
        raise ValueError("Inverse depth has no finite values; check your depth map.")

    # Same clipping as Depth Pro example:
    #   max_invdepth_vizu = min(inverse_depth.max(), 1/0.1)
    #   min_invdepth_vizu = max(1/250, inverse_depth.min())
    max_invdepth_vizu = min(inverse_depth[valid_mask].max(), 1.0 / 0.1)   # 0.1 m
    min_invdepth_vizu = max(1.0 / 250.0, inverse_depth[valid_mask].min()) # 250 m

    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    # Clip to [0, 1] for display
    inverse_depth_normalized = np.clip(inverse_depth_normalized, 0.0, 1.0)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(inverse_depth_normalized, cmap="turbo")
    # plt.colorbar(im, label="Normalized inverse depth")
    # plt.title("DepthPro-style depth visualization")
    plt.axis("off")
    plt.tight_layout()

    # Optionally save the visualization
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def estimate_queue_forward_direction(line_dir_3d: np.ndarray,
                                     orientations_deg: np.ndarray,
                                     inlier_mask: np.ndarray = None) -> tuple[np.ndarray, float]:
    """
    Decide which sign of the RANSAC line direction matches the people's facing directions.

    Angle convention (same as plot_bbox_orientation):
        0°   → arrow up in image  → facing +Z (away from camera)
        90°  → arrow left         → facing -X
        180° → arrow down         → facing -Z (towards camera)
        270° → arrow right        → facing +X

    We map each angle θ to a 3D facing vector in camera coords:
        facing_vec_3d = [-sin θ, 0, cos θ]

    Then we compare that to the line direction in the XZ-plane and
    pick +line_dir_3d or -line_dir_3d based on which aligns better on average.

    Returns:
        queue_forward_dir_3d: (3,) unit 3D vector (either +line_dir_3d or -line_dir_3d).
        score: average alignment dot product (for debugging; >0 means we chose +line_dir_3d).
    """
    # Keep orientations only for inliers if a mask is provided
    if inlier_mask is not None:
        orientations_deg = orientations_deg[inlier_mask]

    if orientations_deg.size == 0:
        # Fallback: no orientations → just use line_dir as-is
        line_dir_3d = line_dir_3d / np.linalg.norm(line_dir_3d)
        return line_dir_3d, 0.0

    # Ensure unit line direction
    line_dir_3d = line_dir_3d / np.linalg.norm(line_dir_3d)

    # Convert degrees → radians
    theta = np.deg2rad(orientations_deg)

    # --- FIXED MAPPING HERE ---
    # 0°   → [ 0, 0, +1]
    # 90°  → [-1, 0,  0]
    # 180° → [ 0, 0, -1]
    # 270° → [+1, 0,  0]
    facing_vecs_3d = np.stack([
        -np.sin(theta),          # X (left/right)
        np.zeros_like(theta),    # Y (ignored)
        np.cos(theta)            # Z (towards/away)
    ], axis=-1)

    # Normalize just in case
    norms = np.linalg.norm(facing_vecs_3d, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    facing_vecs_3d = facing_vecs_3d / norms

    # Compare only in XZ-plane
    line_dir_xz = np.array([line_dir_3d[0], line_dir_3d[2]], dtype=np.float32)
    if np.linalg.norm(line_dir_xz) == 0:
        # Degenerate case: line is vertical in Y; just return the original
        return line_dir_3d, 0.0
    line_dir_xz /= np.linalg.norm(line_dir_xz)

    facing_vecs_xz = facing_vecs_3d[:, [0, 2]]
    norms_xz = np.linalg.norm(facing_vecs_xz, axis=1, keepdims=True)
    norms_xz[norms_xz == 0] = 1.0
    facing_vecs_xz /= norms_xz

    # Dot products: >0 means facing roughly along +line_dir_xz, <0 along -line_dir_xz
    dots = facing_vecs_xz @ line_dir_xz
    score = float(np.mean(dots))

    if score >= 0:
        queue_forward_dir_3d = line_dir_3d
    else:
        queue_forward_dir_3d = -line_dir_3d

    queue_forward_dir_3d = queue_forward_dir_3d / np.linalg.norm(queue_forward_dir_3d)
    return queue_forward_dir_3d, score

def bbox_centers_to_3d(bboxes: np.ndarray,
                       depth: np.ndarray,
                       f_px: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert YOLO bbox centers into 3D points using Depth-Pro metric depth.

    Args:
        bboxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords.
        depth:  (H, W) array of depth in meters, aligned with the RGB image.
        f_px:   focal length in pixels from Depth-Pro (focallength_px).

    Returns:
        points_3d: (M, 3) array of [X, Y, Z] camera-frame coordinates (meters),
                   one per valid bbox center (bboxes that had valid depth).
        centers_uv: (M, 2) array of [u, v] image coordinates (pixels) for the
                   same valid centers.
    """
    H, W = depth.shape
    fx = fy = float(f_px)
    cx = W / 2.0
    cy = H / 2.0

    pts = []
    centers_uv = []

    for (x1, y1, x2, y2) in bboxes:
        # bbox center in pixel coords
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))

        # clamp to image bounds
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        Z = float(depth[v, u])  # depth at center, in meters

        # skip invalid / zero / NaN / inf depths
        if not np.isfinite(Z) or Z <= 0:
            continue

        # back-project to camera coordinates
        # X: right, Y: down, Z: forward
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        pts.append([X, Y, Z])
        centers_uv.append([u, v])

    if len(pts) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32))

    return (np.array(pts, dtype=np.float32),
            np.array(centers_uv, dtype=np.float32))

def project_points_3d_to_2d(points_3d: np.ndarray,
                            depth_shape: tuple[int, int],
                            f_px: float) -> np.ndarray:
    """
    Project 3D camera-frame points [X, Y, Z] back to image coordinates [u, v].

    Args:
        points_3d: (N, 3) array of [X, Y, Z].
        depth_shape: (H, W) of the depth / image.
        f_px: focal length in pixels.

    Returns:
        uv: (K, 2) array of [u, v] for points with Z > 0.
            Points with Z <= 0 are discarded.
    """
    if points_3d.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    H, W = depth_shape
    fx = fy = float(f_px)
    cx = W / 2.0
    cy = H / 2.0

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    # Keep only points in front of the camera
    valid = Z > 0
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32)

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    uv = np.stack([u, v], axis=-1)
    return uv.astype(np.float32)

def visualize_bbox_centers_3d(points_3d: np.ndarray,
                              title: str = "BBox centers in 3D",
                              line_point: np.ndarray = None,
                              line_dir: np.ndarray = None,
                              inlier_mask: np.ndarray = None,
                              queue_forward_dir_3d: np.ndarray = None,
                              image_id: str = "image",
                              depth_shape: tuple[int, int] | None = None,
                              f_px: float | None = None) -> None:
    """
    3D scatter of bbox centers with optional fitted line and oriented arrow.

    Coordinate frame:
        X : right
        Y : down
        Z : forward (camera looks along +Z)

    depth_shape: (H, W) of the depth / image.
    f_px:        focal length in pixels (same as used for back-projection).

    When depth_shape and f_px are given, we:
      - approximate the camera FOV at the median depth of the points
      - set x/y limits and box-aspect so the XY view matches the image aspect.
    """
    if points_3d.size == 0:
        print("No valid 3D points to visualize.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # -------- scatter points ----------
    if inlier_mask is not None:
        inliers = points_3d[inlier_mask]
        outliers = points_3d[~inlier_mask]
        if inliers.size > 0:
            ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2],
                       s=30, label="Inliers")
        if outliers.size > 0:
            ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],
                       s=30, marker='x', label="Outliers")
        # ax.legend()
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=30)

    # -------- fitted line ----------
    arrow_base = None
    if line_point is not None and line_dir is not None:
        diffs = points_3d - line_point
        t_vals = diffs @ line_dir
        t_min, t_max = t_vals.min(), t_vals.max()

        t_line = np.linspace(t_min, t_max, 100)
        line_xyz = line_point[None, :] + t_line[:, None] * line_dir[None, :]

        ax.plot(line_xyz[:, 0], line_xyz[:, 1], line_xyz[:, 2], linewidth=2)

        mid_idx = len(line_xyz) // 2
        arrow_base = line_xyz[mid_idx]

    # -------- queue direction arrow ----------
    if arrow_base is not None and queue_forward_dir_3d is not None:
        arrow_len = 0.5 * np.linalg.norm(
            points_3d.max(axis=0) - points_3d.min(axis=0)
        )
        d = queue_forward_dir_3d / np.linalg.norm(queue_forward_dir_3d)
        ax.quiver(
            arrow_base[0], arrow_base[1], arrow_base[2],
            d[0], d[1], d[2],
            length=arrow_len,
            arrow_length_ratio=0.2,
            color='red'
        )

    ax.set_xlabel("X (m)  (right)")
    ax.set_ylabel("Y (m)  (down)")
    ax.set_zlabel("Z (m)  (forward)")
    # ax.set_title(title)

    # Match image convention: Y increases downward
    ax.invert_yaxis()
    ax.invert_zaxis()

    # -------- set XY limits & aspect WITHOUT using a depth plane --------
    if depth_shape is not None:
        H, W = depth_shape

        # Data-driven XY range
        x_min, x_max = points_3d[:, 0].min(), points_3d[:, 0].max()
        y_min, y_max = points_3d[:, 1].min(), points_3d[:, 1].max()
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Desired aspect: width:height = W:H
        img_aspect = W / float(H)
        if y_range <= 1e-6:
            y_range = 1.0
        current_aspect = x_range / y_range if x_range > 0 else img_aspect

        # Expand the smaller range to match image aspect, keeping center fixed
        if current_aspect < img_aspect:
            # too tall -> widen X
            x_range = img_aspect * y_range
        else:
            # too wide -> expand Y
            y_range = x_range / img_aspect

        x_half = 0.5 * x_range
        y_half = 0.5 * y_range

        ax.set_xlim(x_center - x_half, x_center + x_half)
        ax.set_ylim(y_center + y_half, y_center - y_half)  # inverted Y

        # Full depth range from data
        z_min, z_max = points_3d[:, 2].min(), points_3d[:, 2].max()
        z_range = max(z_max - z_min, 1e-3)

        # Box aspect: match image ratio in XY, keep true Z extent
        ax.set_box_aspect((x_range, y_range, z_range))
    else:
        # Fallback: equal scaling in all axes
        x_range = points_3d[:, 0].ptp()
        y_range = points_3d[:, 1].ptp()
        z_range = points_3d[:, 2].ptp()
        ax.set_box_aspect((x_range, y_range, z_range))

    # -------- view: slightly tilted from camera so depth is visible --------
    ax.view_init(elev=110, azim=-90)

    plt.tight_layout()
    plt.savefig(f"D:\\cv\\LineFinder\\bdoe_res\\{image_id}_3d_scatter.png", dpi=300)
    plt.show()


def ransac_line_3d(points_3d: np.ndarray,
                   num_iters: int = 1000,
                   dist_thresh: float = 0.8,
                   min_inliers_ratio: float = 0.3):
    # print(f"RANSAC line fitting: dist_thresh={dist_thresh}, min_inliers_ratio={min_inliers_ratio}")

    N = points_3d.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 points to fit a 3D line.")

    best_inliers = None
    best_count = 0
    rng = np.random.default_rng()

    for _ in range(num_iters):
        idx = rng.choice(N, size=2, replace=False)
        p1, p2 = points_3d[idx]
        d = p2 - p1
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-6:
            continue
        d = d / d_norm

        diff = points_3d - p1
        dist = np.linalg.norm(np.cross(diff, d), axis=1)
        inliers = dist < dist_thresh
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < max(2, int(min_inliers_ratio * N)):
        print("[WARN] RANSAC could not find a dominant line, using PCA on all points.")
        best_inliers = np.ones(N, dtype=bool)

    # --- PCA refine on inliers ---
    inlier_pts = points_3d[best_inliers]
    center = inlier_pts.mean(axis=0)
    X = inlier_pts - center
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    direction = vh[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # --- IMPORTANT: recompute inliers w.r.t refined line ---
    diff = points_3d - center
    dist = np.linalg.norm(np.cross(diff, direction), axis=1)
    refined_inliers = dist < dist_thresh

    # If recompute killed too many, keep original (avoid degenerate flip)
    if refined_inliers.sum() >= max(2, int(min_inliers_ratio * N)):
        best_inliers = refined_inliers

        # (optional) re-refine once more on refined inliers
        inlier_pts = points_3d[best_inliers]
        center = inlier_pts.mean(axis=0)
        X = inlier_pts - center
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        direction = vh[0]
        direction = direction / (np.linalg.norm(direction) + 1e-8)

    return center, direction, best_inliers


def visualize_queue_on_image(image_path: str,
                             depth: np.ndarray,
                             centers_uv: np.ndarray,
                             points_3d: np.ndarray,
                             line_point: np.ndarray,
                             line_dir: np.ndarray,
                             inlier_mask: np.ndarray,
                             queue_forward_dir_3d: np.ndarray,
                             f_px: float,
                             title: str = "Queue on original image",
                             Image_id: str = "image") -> None:
    """
    Overlay bbox centers, fitted 3D line (projected), and queue direction arrow
    on the original RGB image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}")
        return

    # Convert BGR -> RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W = depth.shape

    # --- 2D positions of the fitted line (projected from 3D) ---
    diffs = points_3d - line_point
    t_vals = diffs @ line_dir
    t_min, t_max = t_vals.min(), t_vals.max()

    t_line = np.linspace(t_min, t_max, 100)
    line_xyz = line_point[None, :] + t_line[:, None] * line_dir[None, :]

    line_uv = project_points_3d_to_2d(line_xyz, depth.shape, f_px)

    # --- Queue direction arrow ---
    arrow_uv_base = None
    arrow_uv_tip = None
    if queue_forward_dir_3d is not None and line_uv.shape[0] > 0:
        # Arrow base at the middle of the 3D line
        t_mid = 0.5 * (t_min + t_max)
        arrow_base_3d = line_point + t_mid * line_dir

        # Arrow length in 3D proportional to the line extent
        line_extent = np.linalg.norm(line_xyz[-1] - line_xyz[0])
        arrow_len_3d = 0.5 * line_extent

        d = queue_forward_dir_3d / np.linalg.norm(queue_forward_dir_3d)
        arrow_tip_3d = arrow_base_3d + arrow_len_3d * d

        arrow_uv = project_points_3d_to_2d(
            np.stack([arrow_base_3d, arrow_tip_3d], axis=0),
            depth.shape,
            f_px
        )
        if arrow_uv.shape[0] == 2:
            arrow_uv_base = arrow_uv[0]
            arrow_uv_tip = arrow_uv[1]

    # --- Plot everything ---
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    ax = plt.gca()

    # Plot bbox centers (valid depth only)
    if centers_uv.size > 0:
        if inlier_mask is not None and inlier_mask.size == centers_uv.shape[0]:
            in_uv = centers_uv[inlier_mask]
            out_uv = centers_uv[~inlier_mask]

            if in_uv.size > 0:
                ax.scatter(in_uv[:, 0], in_uv[:, 1],
                           s=40, c='lime', edgecolors='black', label='Inliers')
            if out_uv.size > 0:
                ax.scatter(out_uv[:, 0], out_uv[:, 1],
                           s=40, c='red', marker='x', label='Outliers')
            ax.legend()
        else:
            ax.scatter(centers_uv[:, 0], centers_uv[:, 1],
                       s=40, c='cyan', edgecolors='black')

    # Plot projected line
    if line_uv.shape[0] > 1:
        ax.plot(line_uv[:, 0], line_uv[:, 1], linewidth=2, color='yellow')

    # Plot arrow for queue direction
    if arrow_uv_base is not None and arrow_uv_tip is not None:
        ax.annotate(
            "",
            xy=(arrow_uv_tip[0], arrow_uv_tip[1]),
            xytext=(arrow_uv_base[0], arrow_uv_base[1]),
            arrowprops=dict(arrowstyle="->", color='orange', linewidth=3)
        )

    # ax.set_title(title)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])  # invert y-axis to match image coords
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"D:\\cv\\LineFinder\\bdoe_res\\{Image_id}_queue.png", dpi=300)
    plt.show()

def classify_queue_facing_direction(queue_forward_dir_3d: np.ndarray,
                                    front_back_tolerance_deg: float = 45.0,
                                    side_x_threshold: float = 0.2) -> str:
    """
    Classify whether the queue is facing towards, away, or sideways
    relative to the camera, using the correct camera frame:

        X : right
        Y : down (image vertical)
        Z : depth forward (camera looks along +Z)

    We:
      1) Project queue_forward_dir_3d onto the XZ plane (ignore Y).
      2) Compute the yaw angle between this 2D vector and +Z.
      3) Use that angle to decide:
           - 'away'    : facing roughly along +Z (into the scene)
           - 'towards' : facing roughly along -Z (toward the camera)
           - 'sideways-left'  / 'sideways-right' / 'sideways'
    """

    q = np.asarray(queue_forward_dir_3d, dtype=np.float32)
    if np.linalg.norm(q) < 1e-6:
        return "unknown"
    q = q / np.linalg.norm(q)

    # ---- Project onto XZ plane (horizontal ground plane) ----
    q_xz = np.array([q[0], q[2]], dtype=np.float32)  # [X, Z]
    if np.linalg.norm(q_xz) < 1e-6:
        # Direction is almost purely vertical; no meaningful yaw
        return "unknown"
    q_xz /= np.linalg.norm(q_xz)

    # Camera forward in XZ plane is +Z
    cam_forward_xz = np.array([0.0, 1.0], dtype=np.float32)  # [0, +1] = +Z

    # Yaw angle between queue direction and camera forward (+Z)
    cos_angle = float(np.dot(q_xz, cam_forward_xz))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    print("queue angle vs camera forward (+Z, XZ plane):", angle_deg, "deg")

    # --- Front / back classification ---
    # Small angle  (~0°)   => queue ≈ +Z  -> facing AWAY
    # Angle ~180°          => queue ≈ -Z  -> facing TOWARDS camera
    if angle_deg <= front_back_tolerance_deg:
        return "away"      # queue_forward_dir ≈ +Z
    elif angle_deg >= 180.0 - front_back_tolerance_deg:
        return "towards"   # queue_forward_dir ≈ -Z

    # --- Sideways: decide left vs right using X component ---
    # q_xz[0] > 0 : pointing toward camera-right
    if q_xz[0] > side_x_threshold:
        return "sideways-right"
    elif q_xz[0] < -side_x_threshold:
        return "sideways-left"
    else:
        return "sideways"



# -------------------------------------------------------------------
# Example usage for ONE image
# -------------------------------------------------------------------
if __name__ == "__main__":

    # Image_id="IMG_5954"
    # Image_id="image_024"
    Image_id="IMG_5955"
    # Image_id="IMG_5956"

    depth_npy_path = "D:\\cv\\LineFinder\\depth_map\\"+Image_id+".npy"
    bboxes_npy_path = "D:\\cv\\LineFinder\\bbox_orient\\"+Image_id+"_bboxes.npy"
    orient_npy_path = "D:\\cv\\LineFinder\\bbox_orient\\"+Image_id+"_orient.npy"
    image_path = "D:\\cv\\LineFinder\\queue_images\\" + Image_id + ".JPG"  # <-- adjust folder

    # focallength_px = 617.5899047851562  # from Depth-Pro for this image 
    # focallength_px = 3187.853759765625
    focallength_px = 441.6351623535156
    # focallength_px =  676.091796875

    visualize_depth_map(depth_npy_path, save_path="D:\\cv\\LineFinder\\depth_map\\"+Image_id+"_depth_viz.jpg")

    depth = load_depth_npy(depth_npy_path)            # your existing function
    bboxes = np.load(bboxes_npy_path).astype(np.float32)
    orientations_deg = load_orientations_npy(orient_npy_path)

    points_3d, centers_uv = bbox_centers_to_3d(bboxes, depth, focallength_px)
    # print("Num 3D points:", points_3d.shape[0])
    # print("Bounding box of 3D points (min/max):")
    # print(points_3d.min(axis=0), points_3d.max(axis=0))

    if points_3d.shape[0] < 2:
        print("Not enough 3D points to fit a line.")
        visualize_bbox_centers_3d(points_3d, title="No line")
    else:
        line_point, line_dir, inlier_mask = ransac_line_3d(
            points_3d,
            num_iters=1000,
            dist_thresh=0.8,
            min_inliers_ratio=0.3,
        )

        queue_forward_dir_3d, score = estimate_queue_forward_direction(
            line_dir_3d=line_dir,
            orientations_deg=orientations_deg,
            inlier_mask=inlier_mask,
        )
        print("Average alignment score (dot product):", score)
        
        facing_label = classify_queue_facing_direction(queue_forward_dir_3d)
        print("Queue is facing:", facing_label)

        visualize_bbox_centers_3d(points_3d,
                          title="Queue with oriented line",
                          line_point=line_point,
                          line_dir=line_dir,
                          inlier_mask=inlier_mask,
                          queue_forward_dir_3d=queue_forward_dir_3d,
                          image_id=Image_id,
                          depth_shape=depth.shape,
                          f_px=focallength_px)
    
        
        visualize_queue_on_image(
            image_path=image_path,
            depth=depth,
            centers_uv=centers_uv,
            points_3d=points_3d,
            line_point=line_point,
            line_dir=line_dir,
            inlier_mask=inlier_mask,
            queue_forward_dir_3d=queue_forward_dir_3d,
            f_px=focallength_px,
            title="Queue line and direction on original image",
            Image_id=Image_id
        )
