import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

# ------------------ Settings ------------------

image_path = "airport_queue/airport_queue.png"
depth_map_path = "airport_queue/airport_queue.tif"
alphapose_json_path = "airport_queue/alphapose-results-airport-queue.json"
output_path = "airport_queue/"

# image_path = "/vast/ds5725/linefinder/LineFinder/Images/Images_By_Us/image_024.jpg"
# depth_map_path = "image_024.tif"
# alphapose_json_path = "alphapose-results-image024.json"
# output_path = "image_024/"

# image_path = "queue_images/multiple_queue.png"
# depth_map_path="multi.tif"
# alphapose_json_path = "alphapose-results-multi.json"
# output_path = "multi_queue/"
# ------------------ Load Data ------------------

pil_img = Image.open(image_path).convert("RGB")
image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
original_height, original_width = image.shape[:2]

depth_map = tiff.imread(depth_map_path)
min_depth = np.min(depth_map)
max_depth = np.max(depth_map)

with open(alphapose_json_path, 'r') as f:
    alphapose_data = json.load(f)

COCO_BODY_PARTS = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]

# ------------------ Functions ------------------

def parse_keypoints(person_dict):
    keypoints = person_dict['keypoints']
    parsed = {}
    for idx, part_name in enumerate(COCO_BODY_PARTS):
        x = keypoints[idx * 3]
        y = keypoints[idx * 3 + 1]
        conf = keypoints[idx * 3 + 2]
        parsed[part_name] = (x, y, conf)
    return parsed

def pixel_to_camera(x, y, depth):
    focal_length = 1000  # Assumed focal length
    cx = original_width / 2
    cy = original_height / 2
    fx = fy = focal_length
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def estimate_facing_direction(keypoints_3d):
    def rotate_ccw_90_topdown(vec):
        x, y, z = vec
        # Rotate 90° counter-clockwise in X-Z plane:
        # [x', z'] = [-z, x]
        # return normalize(np.array([z, y, -x]))
        return normalize(np.array([x, y, z]))

    """Estimate facing direction using nose and shoulder midpoint."""
    if 'Nose' in keypoints_3d and 'LEar' in keypoints_3d and 'REar' in keypoints_3d:
        nose = keypoints_3d['Nose']
        lshoulder = keypoints_3d['LEar']
        rshoulder = keypoints_3d['REar']
        shoulder_mid = (lshoulder + rshoulder) / 2
        facing_vec = shoulder_mid - nose
        return rotate_ccw_90_topdown(facing_vec)

    elif 'LEye' in keypoints_3d and 'REye' in keypoints_3d:
        leye = keypoints_3d['LEye']
        reye = keypoints_3d['REye']
        eye_mid = (leye + reye) / 2
        facing_vec = reye - leye  # Still ambiguous; better with nose
        return rotate_ccw_90_topdown(facing_vec)

    elif 'LShoulder' in keypoints_3d and 'RShoulder' in keypoints_3d:
        shoulder_mid = (keypoints_3d['LShoulder'] + keypoints_3d['RShoulder']) / 2
        facing_vec = keypoints_3d['RShoulder'] - keypoints_3d['LShoulder']
        return rotate_ccw_90_topdown(facing_vec)

    else:
        return None

# ------------------ Process Keypoints ------------------

centers_2d = []
depths = []
facing_vectors = []

for person in alphapose_data:
    keypoints_2d = parse_keypoints(person)
    keypoints_3d = {}

    for part, (x, y, conf) in keypoints_2d.items():
        if conf < 0.5:
            continue
        x_int = int(round(x))
        y_int = int(round(y))
        if 0 <= x_int < original_width and 0 <= y_int < original_height:
            depth = depth_map[y_int, x_int]
            if depth > 0:
                keypoints_3d[part] = pixel_to_camera(x, y, depth)

    if len(keypoints_3d) == 0:
        continue

    # Center point = Nose or middle of Shoulders
    if 'LShoulder' in keypoints_2d and 'RShoulder' in keypoints_2d:
        lx, ly, _ = keypoints_2d['LShoulder']
        rx, ry, _ = keypoints_2d['RShoulder']
        center_2d = ((lx + rx) / 2, (ly + ry) / 2)
    elif 'Nose' in keypoints_3d:
        center_2d = keypoints_2d['Nose'][:2]
    else:
        continue

    x_norm = center_2d[0] / original_width
    y_norm = center_2d[1] / original_height

    centers_2d.append((x_norm, y_norm))
    depths.append(depth_map[int(round(center_2d[1])), int(round(center_2d[0]))])

    # facing_vec = -estimate_facing_direction(keypoints_3d)
    vec = estimate_facing_direction(keypoints_3d)
    if vec is None:
        facing_vec = np.array([0,0,0])   # no arrow
        print("No facing vector could be estimated for a person.")
    else:
        facing_vec = -vec
    facing_vectors.append(facing_vec)


def camera_to_pixel(X, Y, Z):
    focal_length = 1000  # Same as used before
    cx = original_width / 2
    cy = original_height / 2
    fx = fy = focal_length
    x = X * fx / Z + cx
    y = Y * fy / Z + cy
    return x, y

import matplotlib.cm as cm
from matplotlib.colors import to_hex

# Generate unique color per person
cmap = cm.get_cmap('tab20')
point_colors = [to_hex(cmap(i % 20)) for i in range(len(centers_2d))]


# # Overlay on Original Image
# fig2, ax2 = plt.subplots(figsize=(10, 6))
# ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# for i, (x_norm, y_norm) in enumerate(centers_2d):
#     cx = int(x_norm * original_width)
#     cy = int(y_norm * original_height)
#     ax2.scatter(cx, cy, color=point_colors[i], s=40, edgecolor='black')

#     # Draw arrow from shoulder midpoint to nose (if available)
#     person = alphapose_data[i]
#     keypoints_2d = parse_keypoints(person)
    
#     if all(k in keypoints_2d for k in ['LShoulder', 'RShoulder', 'Nose']):
#         lx, ly, lc = keypoints_2d['LShoulder']
#         rx, ry, rc = keypoints_2d['RShoulder']
#         nx, ny, nc = keypoints_2d['Nose']
#         if lc > 0.5 and rc > 0.5 and nc > 0.5:
#             mx, my = (lx + rx) / 2, (ly + ry) / 2  # Midpoint of shoulders
#             dx, dy = nx - mx, ny - my
#             ax2.arrow(mx, my, dx, dy, color='red', head_width=5, head_length=8, length_includes_head=True)

# ax2.set_title('Centers + Shoulder→Nose Facing Arrows')
# ax2.axis('off')
# plt.savefig(output_path+"overlay_with_facing_arrows.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()


# Top-Down X-Z Plane
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_xlim(-0.2, 1)

all_z = [(depth - min_depth) / (max_depth - min_depth) for depth in depths]
z_min_final = max(0, min(all_z) - 0.1)
z_max_final = min(0.5, max(all_z) + 0.1)
ax1.set_ylim(-0.2, z_max_final)
ax1.set_aspect('equal')

for i, (x_norm, _) in enumerate(centers_2d):
    z = all_z[i]
    ax1.scatter(x_norm, z, color=point_colors[i], marker='o')
    facing_vec = facing_vectors[i]
    if facing_vec is not None:
        dx, dz = facing_vec[0], -facing_vec[2]
        norm = np.linalg.norm([dx, dz])
        if norm > 0:
            dx /= norm
            dz /= norm
        arrow_length = 0.03
        ax1.arrow(x_norm, z, dx * arrow_length, dz * arrow_length,
                  head_width=0.02, head_length=0.03, fc='black', ec='black')

ax1.set_xlabel('Normalized X (left to right on image)')
ax1.set_ylabel('Normalized Depth (closer → bottom, farther → top)')
ax1.set_title('Top-Down 2D Human Locations and Facing Directions (X-Z plane)')
plt.savefig(output_path+"topdown_2d_xz_plot_colorcoded.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

# Overlay on Original Image
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
for i, (x_norm, y_norm) in enumerate(centers_2d):
    cx = int(x_norm * original_width)
    cy = int(y_norm * original_height)
    ax2.scatter(cx, cy, color=point_colors[i], s=40, edgecolor='black')

ax2.set_title('Centers Overlaid on Original Image')
ax2.axis('off')
plt.savefig(output_path+"overlay_on_image.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor

# --- prepare data ---
X = np.array([c[0] for c in centers_2d]).reshape(-1,1)
y = np.array(all_z)

# RANSAC Fit
poly_deg = 3
pipeline = make_pipeline(
    PolynomialFeatures(poly_deg, include_bias=False),
    LinearRegression()
)
ransac = RANSACRegressor(
    estimator=pipeline,
    min_samples=max(3, int(len(X) * 0.5)),
    residual_threshold=0.02,
    random_state=42
)
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_.copy()
outlier_mask = ~inlier_mask.copy()

# Remove large-gap outliers
inlier_idx = np.where(inlier_mask)[0]
xi = X[inlier_idx].flatten()
yi = y[inlier_idx]
order = np.argsort(xi)
sorted_idx = inlier_idx[order]
xi_s = xi[order]
yi_s = yi[order]

dx = np.diff(xi_s)
dz = np.diff(yi_s)
dists = np.hypot(dx, dz)
dist_thresh = np.median(dists) * 2.0

bad = []
for j in range(len(dists)):
    if dists[j] > dist_thresh:
        bad.append(sorted_idx[j])

for idx in bad:
    inlier_mask[idx] = False
    outlier_mask[idx] = True

# Re-include close outliers
def point_to_segment_dist(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    t = (vx * wx + vy * wy) / (vx * vx + vy * vy)
    t = np.clip(t, 0.0, 1.0)
    proj_x, proj_y = x1 + t * vx, y1 + t * vy
    return np.hypot(px - proj_x, py - proj_y)

inlier_idx = np.where(inlier_mask)[0]
xi = X[inlier_idx].flatten()
yi = y[inlier_idx]
order = np.argsort(xi)
xi_s = xi[order]
yi_s = yi[order]

dist_to_path_thresh = 0.03
outlier_idx = np.where(outlier_mask)[0]
to_add = []
for idx in outlier_idx:
    px, py = X[idx, 0], y[idx]
    dmin = np.inf
    for i in range(len(xi_s) - 1):
        d = point_to_segment_dist(px, py, xi_s[i], yi_s[i], xi_s[i+1], yi_s[i+1])
        if d < dmin:
            dmin = d
    if dmin < dist_to_path_thresh:
        to_add.append(idx)

for idx in to_add:
    inlier_mask[idx] = True
    outlier_mask[idx] = False

# Final reordering for plot
inlier_idx = np.where(inlier_mask)[0]
xi = X[inlier_idx].flatten()
yi = y[inlier_idx]
order = np.argsort(xi)
xi_s = xi[order]
yi_s = yi[order]

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.2, 1)
ax.set_ylim(min(y) - 0.05, max(y) + 0.05)
ax.set_aspect('equal')

ax.scatter(X[outlier_mask], y[outlier_mask], color='lightgray', marker='o', label='outliers')
ax.scatter(X[inlier_mask], y[inlier_mask], color='tab:blue', s=40, label='inliers')
ax.plot(xi_s, yi_s, color='black', linewidth=2, label='queue')

ax.set_xlabel('Normalized X (left to right on image)')
ax.set_ylabel('Normalized Depth (closer → bottom)')
ax.set_title('Detected Queue with RANSAC + Distance Cleanup')
ax.legend(loc='upper left')

plt.tight_layout()

plt.savefig(output_path + "queue_final.png", bbox_inches='tight', dpi=300)
# plt.show()