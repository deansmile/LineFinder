from lf_paths import LINEFINDER_ROOT, lf_path  # LINEFINDER_ROOT env var
import os
import csv
import base64
import argparse

import cv2
import numpy as np
from openai import OpenAI

from identify_queue_start_end import identify_start_end_bboxes, load_fpx_from_txt


def create_openai_client(api_key: str | None = None) -> OpenAI:
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def normalize_yes_no(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"yes", "true", "1"}:
        return "yes"
    if s in {"no", "false", "0"}:
        return "no"
    return s


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


def ask_gpt5_describe_person(
    client: OpenAI,
    img_b64_png: str,
    role_label: str,
    model: str = "gpt-5.4-mini-high",
) -> str:
    prompt = (
        "Describe this person in a few words focusing on visible appearance such as clothing, hairstyle, accessories."
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_b64_png}",
                    },
                ],
            }
        ],
    )

    return (response.output_text or "").strip()


def read_visibility_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            image_path = str(row.get("image_path", "")).strip()
            status = str(row.get("status", "")).strip().lower()

            if not image_id or not image_path:
                continue
            if status != "ok":
                continue

            rows.append(
                {
                    "source_csv": os.path.basename(csv_path),
                    "image_id": image_id,
                    "image_path": image_path,
                    "start_of_line_visible": normalize_yes_no(row.get("start_of_line_visible", "")),
                    "end_of_line_visible": normalize_yes_no(row.get("end_of_line_visible", "")),
                }
            )
    return rows


def read_all_visibility_csvs(csv_paths: list[str]) -> list[dict]:
    all_rows = []
    seen = set()

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"[WARN] CSV does not exist, skipping: {csv_path}")
            continue

        rows = read_visibility_csv(csv_path)
        print(f"Loaded {len(rows)} valid rows from: {csv_path}")

        for row in rows:
            key = (row["image_id"], row["image_path"])
            if key in seen:
                print(f"[WARN] Duplicate row skipped: {key}")
                continue
            seen.add(key)
            all_rows.append(row)

    return all_rows


def process_one_image(row_in, args, client: OpenAI):
    image_id = row_in["image_id"]
    image_path = row_in["image_path"]
    start_visible = row_in["start_of_line_visible"] == "yes"
    end_visible = row_in["end_of_line_visible"] == "yes"

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
    )

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    start_desc = ""
    end_desc = ""

    if start_visible:
        start_bbox = res["start_bbox_xyxy"]
        start_crop = crop_bbox(img_bgr, start_bbox, pad=args.pad)
        start_b64 = encode_bgr_crop_as_base64_png(start_crop)
        start_desc = ask_gpt5_describe_person(
            client=client,
            img_b64_png=start_b64,
            role_label="START (head)",
            model=args.model,
        )

    if end_visible:
        end_bbox = res["end_bbox_xyxy"]
        end_crop = crop_bbox(img_bgr, end_bbox, pad=args.pad)
        end_b64 = encode_bgr_crop_as_base64_png(end_crop)
        end_desc = ask_gpt5_describe_person(
            client=client,
            img_b64_png=end_b64,
            role_label="END (tail)",
            model=args.model,
        )

    return {
        "source_csv": row_in["source_csv"],
        "image_id": image_id,
        "image_path": image_path,
        "start_of_line_visible": row_in["start_of_line_visible"],
        "end_of_line_visible": row_in["end_of_line_visible"],
        "start_description": start_desc,
        "end_description": end_desc,
        "status": "ok",
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=LINEFINDER_ROOT)
    ap.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--pad", type=int, default=8, help="Padding around bbox crop (pixels)")
    ap.add_argument(
        "--input_csvs",
        nargs="+",
        default=[
            lf_path("code/OL_line_visibility_new.csv"),
            lf_path("code/OLD_line_visibility_new.csv"),
        ],
    )
    ap.add_argument(
        "--output_csv",
        default=lf_path("code/start_end_person_descriptions_new.csv"),
    )
    args = ap.parse_args()

    client = create_openai_client(api_key=args.api_key)

    rows = read_all_visibility_csvs(args.input_csvs)
    print(f"Total valid rows to process: {len(rows)}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    fieldnames = [
        "source_csv",
        "image_id",
        "image_path",
        "start_of_line_visible",
        "end_of_line_visible",
        "start_description",
        "end_description",
        "status",
        "error",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row_in in enumerate(rows, start=1):
            print(f"[{idx}/{len(rows)}] Processing: {row_in['image_path']}")
            try:
                row = process_one_image(row_in, args, client)
                print("  START:", row["start_description"])
                print("  END:  ", row["end_description"])
            except Exception as e:
                row = {
                    "source_csv": row_in["source_csv"],
                    "image_id": row_in["image_id"],
                    "image_path": row_in["image_path"],
                    "start_of_line_visible": row_in["start_of_line_visible"],
                    "end_of_line_visible": row_in["end_of_line_visible"],
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