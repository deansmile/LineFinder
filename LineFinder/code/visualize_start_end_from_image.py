import os
import argparse

from identify_queue_start_end import (
    identify_start_end_bboxes,
    load_fpx_from_txt,
    visualize_start_end_on_image,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True, help="Absolute path to the image")
    ap.add_argument("--root", default="/scratch/ds5725/linefinder/LineFinder",
                    help="Root folder that contains depth_map/, bbox_orient/, focal_length_px.txt")
    ap.add_argument("--out", default=None, help="Output image path (jpg/png). Default: <image_id>_start_end.jpg")
    ap.add_argument("--depth_dir", default=None, help="Optional override for depth_map dir")
    ap.add_argument("--bbox_dir", default=None, help="Optional override for bbox_orient dir")
    ap.add_argument("--fpx_txt", default=None, help="Optional override for focal_length_px.txt")
    args = ap.parse_args()

    image_path = args.image_path
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image_path not found: {image_path}")

    image_id = os.path.splitext(os.path.basename(image_path))[0]

    depth_dir = args.depth_dir or os.path.join(args.root, "depth_map")
    bbox_dir  = args.bbox_dir  or os.path.join(args.root, "bbox_orient")
    fpx_txt   = args.fpx_txt   or os.path.join(args.root, "focal_length_px.txt")

    depth_path  = os.path.join(depth_dir, f"{image_id}.npy")
    bbox_path   = os.path.join(bbox_dir,  f"{image_id}_bboxes.npy")
    orient_path = os.path.join(bbox_dir,  f"{image_id}_orient.npy")

    for p in [depth_path, bbox_path, orient_path, fpx_txt]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    f_px = load_fpx_from_txt(fpx_txt, image_id)

    res = identify_start_end_bboxes(
        image_path=image_path,
        depth_npy_path=depth_path,
        bboxes_npy_path=bbox_path,
        orient_npy_path=orient_path,
        f_px=f_px,
    )

    out_path = args.out or f"{image_id}_start_end.jpg"

    visualize_start_end_on_image(
        image_path=image_path,
        start_bbox=res["start_bbox_xyxy"],  # START (head)
        end_bbox=res["end_bbox_xyxy"],      # END (tail)
        out_path=out_path,
    )

    print("Saved:", os.path.abspath(out_path))
    print("start_orig_idx:", res["start_orig_idx"], "start_bbox:", res["start_bbox_xyxy"].tolist())
    print("end_orig_idx:",   res["end_orig_idx"],   "end_bbox:",   res["end_bbox_xyxy"].tolist())

if __name__ == "__main__":
    main()