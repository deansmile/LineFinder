import os
import argparse

from identify_queue_start_end import (
    identify_start_end_bboxes,
    load_fpx_from_txt,
    visualize_start_end_on_image,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="Folder containing input images")
    ap.add_argument(
        "--root",
        default="/mnt/NAS/home/ds5725/linefinder",
        help="Root folder that contains depth_map/, bbox_orient/, focal_length_px.txt",
    )
    ap.add_argument("--out_dir", required=True, help="Folder to save output visualizations")
    ap.add_argument("--depth_dir", default=None, help="Optional override for depth_map dir")
    ap.add_argument("--bbox_dir", default=None, help="Optional override for bbox_orient dir")
    ap.add_argument("--fpx_txt", default=None, help="Optional override for focal_length_px.txt")
    args = ap.parse_args()

    image_dir = args.image_dir
    out_dir = args.out_dir

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    os.makedirs(out_dir, exist_ok=True)

    depth_dir = args.depth_dir or os.path.join(args.root, "depth_map")
    bbox_dir = args.bbox_dir or os.path.join(args.root, "bbox_orient")
    fpx_txt = args.fpx_txt or os.path.join(args.root, "focal_length_px.txt")

    if not os.path.isfile(fpx_txt):
        raise FileNotFoundError(f"Missing focal length file: {fpx_txt}")

    image_files = sorted(
        [f for f in os.listdir(image_dir) if is_image_file(f)]
    )

    if not image_files:
        print(f"No image files found in: {image_dir}")
        return

    success_count = 0
    skip_count = 0

    for fname in image_files:
        image_path = os.path.join(image_dir, fname)
        image_id = os.path.splitext(fname)[0]

        depth_path = os.path.join(depth_dir, f"{image_id}.npy")
        bbox_path = os.path.join(bbox_dir, f"{image_id}_bboxes.npy")
        orient_path = os.path.join(bbox_dir, f"{image_id}_orient.npy")

        missing_files = [
            p for p in [depth_path, bbox_path, orient_path] if not os.path.isfile(p)
        ]
        if missing_files:
            print(f"[SKIP] {image_id} - missing required file(s):")
            for p in missing_files:
                print(f"       {p}")
            skip_count += 1
            continue

        try:
            f_px = load_fpx_from_txt(fpx_txt, image_id)

            res = identify_start_end_bboxes(
                image_path=image_path,
                depth_npy_path=depth_path,
                bboxes_npy_path=bbox_path,
                orient_npy_path=orient_path,
                f_px=f_px,
            )

            out_path = os.path.join(out_dir, f"{image_id}_start_end.jpg")

            visualize_start_end_on_image(
                image_path=image_path,
                start_bbox=res["start_bbox_xyxy"],  # START
                end_bbox=res["end_bbox_xyxy"],      # END
                out_path=out_path,
            )

            print(f"[OK] {image_id}")
            print(f"     Saved: {out_path}")
            print(f"     start_orig_idx: {res['start_orig_idx']}, start_bbox: {res['start_bbox_xyxy'].tolist()}")
            print(f"     end_orig_idx:   {res['end_orig_idx']}, end_bbox:   {res['end_bbox_xyxy'].tolist()}")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] {image_id}: {e}")
            skip_count += 1

    print("\nDone.")
    print(f"Successful: {success_count}")
    print(f"Skipped/Errored: {skip_count}")


if __name__ == "__main__":
    main()