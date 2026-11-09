import os

folders = [
    "/scratch/ds5725/linefinder/LineFinder/Images/QueuesInSupermarketNew",
    "/scratch/ds5725/linefinder/LineFinder/Images/QueuesInThemeParks",
    "/scratch/ds5725/linefinder/LineFinder/Images/QueuesOutdoors",
]

output_file = "olivia_luna_image_paths.txt"

all_paths = []

for folder in folders:
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            full_path = os.path.abspath(os.path.join(dirpath, fname))
            all_paths.append(full_path)

# Sort for deterministic order
all_paths = sorted(all_paths)

with open(output_file, "w") as f:
    for path in all_paths:
        f.write(path + "\n")

print(f"Saved {len(all_paths)} paths to {output_file}")