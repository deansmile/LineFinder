# depthpro_batch.py
import os
import torch
import numpy as np
from matplotlib import pyplot as plt

import depth_pro


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # use gpu if it has more than 7GB total
        if torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024 > 7:
            device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def iter_pngs(folder: str):
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".png"):
            yield fname, os.path.join(folder, fname)


def save_invdepth_png(depth: np.ndarray, out_png: str, title: str):
    """Save a nice inverse-depth visualization."""
    inv = 1.0 / np.clip(depth, 1e-6, None)

    # Clip for nicer contrast (meters)
    near, far = 0.1, 250.0
    inv_min = 1.0 / far
    inv_max = 1.0 / near
    inv = np.clip(inv, inv_min, inv_max)

    inv_norm = (inv - inv_min) / (inv_max - inv_min + 1e-12)

    plt.figure()
    plt.imshow(inv_norm, cmap="turbo")
    plt.colorbar(label="Inverse depth (normalized)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    in_dir = "/scratch/ds5725/OOPS/images_resized"
    out_dir = "/scratch/ds5725/OOPS/depthpro_out"  # change if you want
    os.makedirs(out_dir, exist_ok=True)

    device = get_torch_device()
    print("Using device:", device)

    # Load model and preprocessing transform ONCE
    # Note: keep precision=torch.half for speed on GPU; if CPU, float32 is safer.
    precision = torch.half if device.type == "cuda" else torch.float32
    model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=precision,
    )
    model.eval()

    n = 0
    for fname, image_path in iter_pngs(in_dir):
        n += 1
        image_id = os.path.splitext(fname)[0]

        npy_out = os.path.join(out_dir, f"{image_id}.npy")
        png_out = os.path.join(out_dir, f"{image_id}_invdepth.png")

        try:
            # Load and preprocess an image
            image, _, f_px = depth_pro.load_rgb(image_path)
            image = transform(image)

            # Inference
            with torch.no_grad():
                prediction = model.infer(image, f_px=f_px)

            depth = prediction["depth"].detach().float().cpu().numpy().squeeze()  # [H,W]
            focallength_px = prediction.get("focallength_px", None)

            print(f"[{n}] {image_id}")
            print(f"  depth min/max: {depth.min():.4f} / {depth.max():.4f}")
            if focallength_px is not None:
                try:
                    print(f"  focal length px: {float(focallength_px):.2f}")
                except Exception:
                    print(f"  focal length px: {focallength_px}")

            # Save depth map
            np.save(npy_out, depth.astype(np.float32))
            # Save viz
            save_invdepth_png(depth, png_out, title=f"{image_id} - inverse depth (viz)")

            print(f"  Saved depth: {npy_out}")
            print(f"  Saved viz  : {png_out}")

        except Exception as e:
            print(f"[{n}] ERROR on {image_id}: {repr(e)}")
            continue

    print(f"\nDone. Processed {n} images.")
    print("Outputs in:", out_dir)


if __name__ == "__main__":
    main()