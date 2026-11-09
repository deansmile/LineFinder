import os

# Root folder
root = "/vast/ds5725/linefinder/LineFinder/Images"

# Subfolders
subfolders = ["QueuesInSupermarkets", "QueuesInBanks", "ImagesOnline", "Images_By_Us"]

# Collect all absolute file paths
all_files = []
for sub in subfolders:
    folder_path = os.path.join(root, sub)
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            all_files.append(os.path.abspath(os.path.join(dirpath, f)))

print(f"Total files found: {len(all_files)}")
# Example: print first 10 paths
print(all_files[:10])
