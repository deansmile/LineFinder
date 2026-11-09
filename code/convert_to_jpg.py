import os
from PIL import Image

folder = "/scratch/ds5725/linefinder/LineFinder/Images/QueuesInSupermarketNew"

valid_ext = (".avif", ".jpeg", ".jpg", ".png", ".webp")

# Get all image files sorted
files = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith(valid_ext)
])

print(f"Found {len(files)} images")

counter = 1
new_files = []

for fname in files:
    old_path = os.path.join(folder, fname)
    new_name = f"supermarket_{counter:03d}.jpg"
    new_path = os.path.join(folder, new_name)

    try:
        with Image.open(old_path) as img:
            rgb = img.convert("RGB")  # ensure JPG compatible
            rgb.save(new_path, "JPEG", quality=95)

        new_files.append(new_path)
        counter += 1

    except Exception as e:
        print(f"Failed on {fname}: {e}")

# Remove original files AFTER conversion
for fname in files:
    old_path = os.path.join(folder, fname)
    if not os.path.basename(old_path).startswith("supermarket_"):
        os.remove(old_path)

print("Conversion complete.")