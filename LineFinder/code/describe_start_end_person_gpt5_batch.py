from lf_paths import LINEFINDER_ROOT, lf_path  # LINEFINDER_ROOT env var
import os
try:  # load API keys from the repo-root .env (python-dotenv optional)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
import os
import csv
import base64
import argparse

import cv2
import numpy as np
import openai

from identify_queue_start_end import identify_start_end_bboxes, load_fpx_from_txt


def encode_bgr_crop_as_base64_png(crop_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", crop_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def crop_bbox(img_bgr: np.ndarray, bbox_xyxy: np.ndarray, pad: int = 8) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]

    x1 = int(max(0, np.floor(x1) - pad))
    y1 = int(max(0, np.floor(y1) - pad))
    x2 = int(min(w - 1, np.ceil(x2) + pad))
    y2 = int(min(h - 1, np.ceil(y2) + pad))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop after clamp: {(x1, y1, x2, y2)}")

    return img_bgr[y1:y2, x1:x2].copy()


def ask_gpt5_describe_person(img_b64_png: str, role_label: str, model: str = "gpt-5") -> str:
    system_msg = (
        "You are a vision assistant. Describe the person in the image crop. "
        "Focus on visible appearance such as clothing, hairstyle, accessories, "
        "and pose. Keep it short."
    )

    user_text = (
        f"This is the {role_label} person in a queue. "
        "Describe this person in a few words."
    )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64_png}"},
                    },
                ],
            },
        ],
        max_completion_tokens=300,
    )

    return (resp.choices[0].message.content or "").strip()


def list_all_images(image_dirs):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []

    for d in image_dirs:
        if not os.path.isdir(d):
            print(f"[WARN] Folder does not exist, skipping: {d}")
            continue

        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
                image_paths.append(path)

    return image_paths


def process_one_image(image_path, args):
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    depth_path = os.path.join(args.root, "depth_map", f"{image_id}.npy")
    bbox_path = os.path.join(args.root, "bbox_orient", f"{image_id}_bboxes.npy")
    orient_path = os.path.join(args.root, "bbox_orient", f"{image_id}_orient.npy")
    fpx_txt = os.path.join(args.root, "focal_length_px.txt")

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
        ransac_dist_thresh=args.ransac_dist_thresh,
        ransac_min_inliers_ratio=args.ransac_min_inliers_ratio,
    )

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    start_bbox = res["start_bbox_xyxy"]
    end_bbox = res["end_bbox_xyxy"]

    start_crop = crop_bbox(img_bgr, start_bbox, pad=args.pad)
    end_crop = crop_bbox(img_bgr, end_bbox, pad=args.pad)

    start_b64 = encode_bgr_crop_as_base64_png(start_crop)
    end_b64 = encode_bgr_crop_as_base64_png(end_crop)

    start_desc = ask_gpt5_describe_person(start_b64, role_label="START (head)", model=args.model)
    end_desc = ask_gpt5_describe_person(end_b64, role_label="END (tail)", model=args.model)

    return {
        "image_id": image_id,
        "image_path": image_path,
        "start_bbox_xyxy": [float(x) for x in start_bbox.tolist()],
        "end_bbox_xyxy": [float(x) for x in end_bbox.tolist()],
        "start_description": start_desc,
        "end_description": end_desc,
        "status": "ok",
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=LINEFINDER_ROOT)
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--pad", type=int, default=8, help="Padding around bbox crop (pixels)")
    ap.add_argument("--ransac_dist_thresh", type=float, default=0.8)
    ap.add_argument("--ransac_min_inliers_ratio", type=float, default=0.3)
    ap.add_argument(
        "--output_csv",
        default=lf_path("start_end_person_descriptions.csv"),
    )
    args = ap.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY_PROJ") or os.getenv("OPENAI_API_KEY")

    image_dirs = [
        lf_path("Images/QueuesInSupermarketNew"),
        lf_path("Images/QueuesInThemeParks"),
        lf_path("Images/QueuesOutdoors"),
    ]

    image_paths = list_all_images(image_dirs)
    print(f"Found {len(image_paths)} images.")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    fieldnames = [
        "image_id",
        "image_path",
        "start_bbox_xyxy",
        "end_bbox_xyxy",
        "start_description",
        "end_description",
        "status",
        "error",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, image_path in enumerate(image_paths, start=1):
            print(f"[{idx}/{len(image_paths)}] Processing: {image_path}")
            try:
                row = process_one_image(image_path, args)
                print("  START:", row["start_description"])
                print("  END:  ", row["end_description"])
            except Exception as e:
                row = {
                    "image_id": os.path.splitext(os.path.basename(image_path))[0],
                    "image_path": image_path,
                    "start_bbox_xyxy": "",
                    "end_bbox_xyxy": "",
                    "start_description": "",
                    "end_description": "",
                    "status": "error",
                    "error": str(e),
                }
                print(f"  [ERROR] {e}")

            writer.writerow(row)
            f.flush()

    print(f"Done. Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()