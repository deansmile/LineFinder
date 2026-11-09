import os

gt_folder = "/vast/ds5725/linefinder/LineFinder/GT_json"
img_folder = "/scratch/ds5725/linefinder/LineFinder/Images"
output_txt = "images_without_gt.txt"

# 1️⃣ Collect GT names
gt_names = set()
for dirpath, _, filenames in os.walk(gt_folder):
    for fname in filenames:
        if fname.lower().endswith(".json"):
            gt_names.add(os.path.splitext(fname)[0])

# 2️⃣ Collect image paths
img_paths = {}
for dirpath, _, filenames in os.walk(img_folder):
    for fname in filenames:
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            key = os.path.splitext(fname)[0]
            img_paths[key] = os.path.join(dirpath, fname)

# 3️⃣ Compute missing
missing_keys = sorted(set(img_paths.keys()) - gt_names)

# 4️⃣ Write full paths
with open(output_txt, "w") as f:
    for key in missing_keys:
        f.write(img_paths[key] + "\n")

print(f"Saved {len(missing_keys)} image paths to {output_txt}")