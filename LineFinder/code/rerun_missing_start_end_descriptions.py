import os
try:  # load API keys from the repo-root .env (python-dotenv optional)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
import os
import re
import csv
import base64
import argparse

import cv2
import numpy as np
import openai

from identify_queue_start_end import identify_start_end_bboxes, load_fpx_from_txt
from tqdm import tqdm

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
        ]
    )

    return (resp.choices[0].message.content or "").strip()


def parse_missing_roles_from_slurm(slurm_out_path):
    """
    Parse lines like:
      supermarket_001 -> END visible but description missing
      supermarket_004 -> START visible but description missing
    Returns:
      {
        "supermarket_001": {"START": False, "END": True},
        "supermarket_004": {"START": True, "END": True},
      }
    """
    missing = {}
    pattern = re.compile(r"^\s*([^\s]+)\s*->\s*(START|END)\s+visible but description missing\s*$")

    with open(slurm_out_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            image_id = m.group(1).strip()
            role = m.group(2).strip()

            if image_id not in missing:
                missing[image_id] = {"START": False, "END": False}
            missing[image_id][role] = True

    return missing


def load_existing_descriptions(csv_path):
    rows = []
    by_image_id = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            row = dict(row)
            image_id = row["image_id"].strip()
            rows.append(row)
            by_image_id[image_id] = row

    return rows, by_image_id, fieldnames


def save_updated_descriptions(csv_path, rows, fieldnames):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_image_path(root, image_id):
    candidates = [
        os.path.join(root, "Images", "QueuesInSupermarketNew", f"{image_id}.jpg"),
        os.path.join(root, "Images", "QueuesInSupermarketNew", f"{image_id}.png"),
        os.path.join(root, "Images", "QueuesInThemeParks", f"{image_id}.jpg"),
        os.path.join(root, "Images", "QueuesInThemeParks", f"{image_id}.png"),
        os.path.join(root, "Images", "QueuesOutdoors", f"{image_id}.jpg"),
        os.path.join(root, "Images", "QueuesOutdoors", f"{image_id}.png"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find image file for image_id={image_id}")


def process_missing_sides(image_id, roles_to_process, image_path, args):
    depth_path = os.path.join(args.root, "depth_map", f"{image_id}.npy")
    bbox_path = os.path.join(args.root, "bbox_orient", f"{image_id}_bboxes.npy")
    orient_path = os.path.join(args.root, "bbox_orient", f"{image_id}_orient.npy")
    fpx_txt = os.path.join(args.root, "focal_length_px.txt")

    for p in [image_path, depth_path, bbox_path, orient_path, fpx_txt]:
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

    out = {}

    if roles_to_process.get("START", False):
        start_bbox = res["start_bbox_xyxy"]
        start_crop = crop_bbox(img_bgr, start_bbox, pad=args.pad)
        start_b64 = encode_bgr_crop_as_base64_png(start_crop)
        out["start_bbox_xyxy"] = [float(x) for x in start_bbox.tolist()]
        out["start_description"] = ask_gpt5_describe_person(
            start_b64, role_label="START (head)", model=args.model
        )

    if roles_to_process.get("END", False):
        end_bbox = res["end_bbox_xyxy"]
        end_crop = crop_bbox(img_bgr, end_bbox, pad=args.pad)
        end_b64 = encode_bgr_crop_as_base64_png(end_crop)
        out["end_bbox_xyxy"] = [float(x) for x in end_bbox.tolist()]
        out["end_description"] = ask_gpt5_describe_person(
            end_b64, role_label="END (tail)", model=args.model
        )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/ds5725/linefinder/LineFinder")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--ransac_dist_thresh", type=float, default=0.8)
    ap.add_argument("--ransac_min_inliers_ratio", type=float, default=0.3)

    ap.add_argument(
        "--slurm_out",
        default="/scratch/ds5725/linefinder/LineFinder/code/slurm-4110108.out",
    )
    ap.add_argument(
        "--input_csv",
        default="/scratch/ds5725/linefinder/LineFinder/code/start_end_person_descriptions.csv",
    )
    ap.add_argument(
        "--output_csv",
        default="/scratch/ds5725/linefinder/LineFinder/code/start_end_person_descriptions_rerun.csv",
    )
    args = ap.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY_PROJ") or os.getenv("OPENAI_API_KEY")


    missing_map = parse_missing_roles_from_slurm(args.slurm_out)
    print(f"Found {len(missing_map)} images with missing START/END descriptions in slurm log.")

    rows, by_image_id, fieldnames = load_existing_descriptions(args.input_csv)

    needed_fields = [
        "image_id",
        "image_path",
        "start_bbox_xyxy",
        "end_bbox_xyxy",
        "start_description",
        "end_description",
        "status",
        "error",
    ]
    for f in needed_fields:
        if f not in fieldnames:
            raise ValueError(f"Missing required column in input CSV: {f}")

    processed = 0
    failed = 0

    for image_id, roles in tqdm(missing_map.items()):
        print(f"\nProcessing {image_id} | START={roles['START']} END={roles['END']}")

        row = by_image_id.get(image_id)
        if row is None:
            image_path = infer_image_path(args.root, image_id)
            row = {
                "image_id": image_id,
                "image_path": image_path,
                "start_bbox_xyxy": "",
                "end_bbox_xyxy": "",
                "start_description": "",
                "end_description": "",
                "status": "",
                "error": "",
            }
            rows.append(row)
            by_image_id[image_id] = row
        else:
            image_path = row.get("image_path", "").strip()
            if not image_path or not os.path.isfile(image_path):
                image_path = infer_image_path(args.root, image_id)
                row["image_path"] = image_path

        try:
            result = process_missing_sides(image_id, roles, image_path, args)

            if "start_bbox_xyxy" in result:
                row["start_bbox_xyxy"] = str(result["start_bbox_xyxy"])
            if "end_bbox_xyxy" in result:
                row["end_bbox_xyxy"] = str(result["end_bbox_xyxy"])
            if "start_description" in result:
                row["start_description"] = result["start_description"]
                print("  START:", result["start_description"])
            if "end_description" in result:
                row["end_description"] = result["end_description"]
                print("  END:  ", result["end_description"])

            row["status"] = "ok"
            row["error"] = ""
            processed += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            row["status"] = "error"
            row["error"] = str(e)
            failed += 1
        

    save_updated_descriptions(args.output_csv, rows, fieldnames)

    print("\nDone.")
    print(f"Processed successfully: {processed}")
    print(f"Failed: {failed}")
    print(f"Saved updated CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()