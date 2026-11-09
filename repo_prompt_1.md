Set up git for /scratch/ds5725/linefinder and push it to GitHub as the main branch. Don't modify, move or delete any project files.

1. Survey the folder: top-level layout, total size, the largest files and folders, and which folders hold data, model weights or outputs versus code. Show me before continuing.
2. Scan for secrets (API keys for OpenAI, Gemini, Kimi, Hugging Face tokens, passwords). For each one, show me the file and line. Propose moving keys into a .env file loaded with os.getenv or python-dotenv, and apply that change only after I confirm.
3. Write a .gitignore. Exclude data, images, outputs, results, checkpoints, model weights (*.pt, *.pth, *.ckpt, *.safetensors), archives, .env, __pycache__ and notebook checkpoints. Also exclude any single file over 10 MB. Keep code, configs, small CSV/JSON results and the README.
4. Run git init -b main, stage everything, and show me git status plus the total staged size. Wait for my OK.
5. Commit with the message "Torch snapshot". Add remote origin git@github.com:deansmile/LineFinder.git and push. If the remote isn't empty, stop and tell me what's there.
6. Write a data manifest, data_manifest_torch.csv (relative path, size, mtime, sha256), covering every git-ignored data or output file. Save it outside the repo at /scratch/ds5725/linefinder_manifest_torch.csv.