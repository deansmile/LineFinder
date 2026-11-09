import os
import PIL.Image
import depth_pro
import torch
import numpy as np
from tqdm import tqdm

# ------------- Paths & Config -------------
IMAGE_LIST_PATH = "/scratch/ds5725/linefinder/LineFinder/code/olivia_luna_image_paths.txt"

DEPTH_OUT_DIR = "/scratch/ds5725/linefinder/LineFinder/depth_map"
FOCAL_TXT_PATH = "/scratch/ds5725/linefinder/LineFinder/focal_length_px.txt"

os.makedirs(DEPTH_OUT_DIR, exist_ok=True)


def get_torch_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024 > 7:
            device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def load_image_paths(list_path: str):
    paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p and not p.startswith("#"):
                paths.append(p)
    return paths


def main():
    device = get_torch_device()
    print(f"Using device: {device}")

    # Load model once
    model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=torch.half,
    )
    model.eval()

    # Load all images to process
    image_paths = load_image_paths(IMAGE_LIST_PATH)
    print(f"Found {len(image_paths)} images.")

    # Open the focal length log file once
    with open(FOCAL_TXT_PATH, "a", encoding="utf-8") as focal_log:

        for idx, image_path in tqdm(enumerate(image_paths)):
            if not os.path.isfile(image_path):
                print(f"[WARN] Missing image file: {image_path}")
                continue

            base = os.path.basename(image_path)
            image_id, _ = os.path.splitext(base)

            depth_save_path = os.path.join(DEPTH_OUT_DIR, f"{image_id}.npy")

            # Skip if already processed
            if os.path.exists(depth_save_path):
                print(f"[INFO] ({idx+1}/{len(image_paths)}) Already exists, skipping depth for {image_id}")
                continue

            try:
                # Load and preprocess an image
                image, _, f_px = depth_pro.load_rgb(image_path)
                image = transform(image)

                # Run inference
                prediction = model.infer(image, f_px=f_px)

                # Depth
                depth = prediction["depth"].detach().cpu().numpy().squeeze()

                print(f"\n({idx+1}/{len(image_paths)}) Processing {image_id}")
                print(f"  Saving depth: {depth_save_path}")
                np.save(depth_save_path, depth)

                # Focal length
                focallength_px = float(prediction["focallength_px"])
                focal_log.write(f"{image_id} {focallength_px}\n")
                focal_log.flush()

                print(f"  Focal length px: {focallength_px}")

            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {e}")


if __name__ == "__main__":
    main()
