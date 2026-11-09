Merge the Shannon work into /scratch/ds5725/linefinder, then make Torch the single working copy.

Code:
1. Fetch origin and merge origin/shannon into main with --allow-unrelated-histories.
2. For every conflict, show me a short summary of how the Torch and Shannon versions differ: which is newer (from both manifests' mtimes or git info), and what functionally changed. Recommend which to keep, or how to combine them. Resolve each one only after I confirm.
3. Run a quick check that the main scripts still import and parse: python -m py_compile on all .py files. Then commit, push main, and delete the shannon branch on GitHub.

Data:
4. Download the Hugging Face dataset deansmile123/LineFinder at revision shannon into /scratch/ds5725/linefinder_shannon_staging. Leave the live data folder alone for now.
5. Compare the Shannon manifest against a fresh manifest of the Torch data. Group the files into: Shannon only, Torch only, identical, and different content. For each file in the "different" group, show both mtimes and sizes. Wait for my decision.
6. Copy the Shannon-only files and the different files I approve into the Torch data folder. Don't overwrite anything without my approval.
7. Confirm the dataset is still private, then upload the merged data folder to the main revision with upload_large_folder. Delete the Hugging Face branch shannon once the upload is done.
8. Add a short DATA.md to the repo explaining where data lives (the private HF dataset), how to download it, and that the dataset must not be made public. Commit and push it.
9. Report what you merged, what you skipped, and the final repo and dataset state.