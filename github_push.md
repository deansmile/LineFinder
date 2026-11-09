Push the code in /mnt/NAS/home/ds5725/linefinder to GitHub as a branch named shannon, and upload its data to Hugging Face on a separate branch. Don't modify, move or delete any project files.

1. Survey the folder the same way: layout, size, largest items, and code versus data/weights/outputs. Show me.
2. Scan for secrets. Show me each one and propose moving it into .env. Apply only after I confirm.
3. Fetch https://github.com/deansmile/LineFinder (main) and reuse its .gitignore, adding any Shannon-specific large folders. Run git init -b shannon, stage everything, and show me git status plus the staged size. Wait for my OK.
4. Commit with the message "Shannon snapshot". Add remote origin git@github.com:deansmile/LineFinder.git and push to the branch shannon. Don't touch main.
5. Write data_manifest_shannon.csv (relative path, size, mtime, sha256) for every git-ignored data or output file. Save it to /mnt/NAS/home/ds5725/linefinder_manifest_shannon.csv.
6. Confirm that the dataset deansmile123/LineFinder is private, using huggingface_hub's dataset_info. Stop if it's public.
7. Create the branch "shannon" on that dataset repo and upload the git-ignored data and outputs there with upload_large_folder (repo_type dataset, revision shannon). Leave out model weights that can be re-downloaded (YOLO, DepthPro, JointBDOE) and list them for me. Upload the manifest CSV to the same branch.