import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO

# File paths
image_path = "queue_images/image_024.jpg"
depth_map_path = "image_024.tif"

# Load the original image (without resizing)
image = cv2.imread(image_path)
original_height, original_width = image.shape[:2]

# Load the YOLO model and perform inference (detect humans)
model = YOLO("yolo11n.pt")
results = model(image_path, classes=0)

# Extract bounding boxes in normalized xywh format
tensor_boxes = results[0].boxes.xywhn  # (x_center, y_center, width, height) normalized

# print(dir(results[0]))
# masks = results[0].masks
# xy_coords = masks.xy
# print(len(xy_coords))
# print(xy_coords[0].shape)
# exit()
# Load the depth map
depth_map = tiff.imread(depth_map_path)

# Find min and max depth values for normalization
min_depth = np.min(depth_map)
max_depth = np.max(depth_map)

# Define color bins for normalized X values
color_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = ["red", "orange", "yellow", "green", "blue", "purple"]

# Store the normalized 3D points (X, Y, Z) and corresponding colors
points_3D = []
point_colors = []

# Extract (x, y) centers and get their depth (z)
for box in tensor_boxes:
    x_norm = box[0].item()  # X already normalized
    y_norm = box[1].item()  # Y already normalized

    # Get the corresponding depth value (Z-coordinate)
    x_center = int(box[0] * original_width)  # Convert to image coordinates
    y_center = int(box[1] * original_height)

    if 0 <= y_center < original_height and 0 <= x_center < original_width:
        z_depth = depth_map[y_center, x_center]
    else:
        z_depth = min_depth  # Default to min depth if out of bounds

    # Normalize Z using min-max scaling
    z_norm = (z_depth - min_depth) / (max_depth - min_depth) if max_depth > min_depth else 0

    # Assign color based on the normalized X-coordinate
    for i in range(len(color_bins) - 1):
        if color_bins[i] <= x_norm < color_bins[i + 1]:
            point_color = colors[i]
            break
    else:
        point_color = colors[-1]  # Assign last color if X is exactly 1.0

    # Store (x, y, z) and color
    points_3D.append((x_norm, y_norm, z_norm))
    point_colors.append(point_color)

# Convert to NumPy array for plotting
points_3D = np.array(points_3D)

# Plot the 3D points
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Set the axis limits explicitly from 0 to 1
ax.set_xlim(0, 1)  # X from 0 to 1
ax.set_ylim(0, 1)  # Y from 0 to 1

# Scatter plot with assigned colors
for i in range(len(points_3D)):
    ax.scatter(points_3D[i, 0], points_3D[i, 1], points_3D[i, 2], color=point_colors[i], marker='o')

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# ax.set_title("3D Visualization of Detected Human Centers (Color-Coded by X)")

# Set the viewing angle to mimic a top-down 2D image view
# ax.view_init(elev=-90, azim=-90)  # Top-down view, aligning with image coordinates
# ax.view_init(elev=90, azim=180)
ax.view_init(elev=0, azim=-90)  # Top-down view with Z flipped


# Create a legend for color coding
legend_patches = [
    plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=colors[i], 
               label=f"{color_bins[i]:.1f} ≤ X < {color_bins[i+1]:.1f}")
    for i in range(len(colors)-1)  # Loop over the color list, not color_bins
]
# plt.legend(handles=legend_patches, title="Normalized X Color Bins", loc="lower left")

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.axis('off')

# Save the image
plt.savefig("image_024_topdown.png", bbox_inches='tight', pad_inches=0, dpi=300)

plt.show()
