import PIL.Image
import depth_pro
import torch, cv2
import numpy as np
from matplotlib import pyplot as plt

def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # use gpu if it has more than 7GB free
        if torch.cuda.get_device_properties(0).total_memory/1024/1024/1024 > 7:
            device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

image_id="WheelieBin_Set1_Pos4_OOPS0.5"
image_path="/scratch/ds5725/OOPS/images_resized/"+image_id+".png"
save_path=image_id+".npy"
# image_path="/vast/ds5725/linefinder/ml-depth-pro/image_024.jpg"
# image_path="/scratch/ds5725/LineFinder/ml-depth-pro/data/example.jpg"
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
# depth = prediction["depth"]  # Depth in [m].


depth = prediction["depth"].detach().cpu().numpy().squeeze()
print(image_id)
print(f"Min depth value: {np.min(depth)}")
print(f"Max depth value: {np.max(depth)}")
print(f"Saving depth map to: {str(save_path)}")
np.save(save_path, depth)

focallength_px = prediction["focallength_px"]
print(f"Focal length in pixels: {focallength_px}")

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
plt.title(f"{image_id} - inverse depth (viz)")
plt.axis("off")
plt.tight_layout()

png_out = image_id + "_invdepth.png"
plt.savefig(png_out, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", png_out)