from PIL import Image
import numpy as np

def remove_white_border_and_resize(input_path, output_path, size=(512, 512), threshold=240):
    # Load image (keep RGBA if present)
    img = Image.open(input_path)
    img_np = np.array(img)

    # Handle grayscale images
    if len(img_np.shape) == 2:
        mask = img_np < threshold
    else:
        # Use RGB channels only
        rgb = img_np[..., :3]

        # Mask for non-white pixels (robust to near-white)
        mask = np.any(rgb < threshold, axis=-1)

    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        print(f"{input_path} is fully white, skipping")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop
    cropped = img_np[y_min:y_max+1, x_min:x_max+1]

    # Convert back to PIL
    cropped_img = Image.fromarray(cropped)

    # Resize
    resized_img = cropped_img.resize(size, Image.BILINEAR)

    # Save
    resized_img.save(output_path)


# Example usage
remove_white_border_and_resize(
    "outdoor_021_queue.png",
    "outdoor_021.png",
    size=(2000, 1333)
)