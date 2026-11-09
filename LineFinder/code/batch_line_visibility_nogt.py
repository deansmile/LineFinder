from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
import csv
import numpy as np
import cv2

from identify_queue_start_end import identify_start_end_bboxes, load_fpx_from_txt

# ---------- INPUTS ----------
IMAGE_LIST_TXT = lf_path("image_paths.txt")

DEPTH_DIR = lf_path("depth_map")
BBOX_ORIENT_DIR = lf_path("bbox_orient")
FOCAL_TXT = lf_path("focal_length_px.txt")

OUTPUT_CSV = "line_visibility_new.csv"


# ---------- Helper ----------
def read_image_list(txt_path):
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def _bbox_edge_flags_horizontal(bbox_xyxy, W):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    bbox_w = max(x2 - x1, 1.0)

    dist_left = x1
    dist_right = (W - 1) - x2

    near_left = dist_left < bbox_w
    near_right = dist_right < bbox_w
    touches_any = near_left or near_right

    return touches_any, near_left, near_right


def _location_bucket_from_center_x(cx, W):
    r = cx / max(W, 1)
    if r < 0.2:
        return "far left"
    elif r < 0.4:
        return "center left"
    elif r < 0.6:
        return "center"
    elif r < 0.8:
        return "center right"
    else:
        return "far right"


def endpoint_fields(bbox_xyxy, W, H=None, margin_px=None):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    cx = 0.5 * (x1 + x2)

    touches_any, near_left, near_right = _bbox_edge_flags_horizontal(bbox_xyxy, W)

    if touches_any:
        visible = "no"
        location = "N/A"
        turn = "left" if near_left else "right"
    else:
        visible = "yes"
        location = _location_bucket_from_center_x(cx, W)
        turn = "N/A"

    return visible, location, turn

def process_one_image(img_path):
    image_id = os.path.splitext(os.path.basename(img_path))[0]

    depth_path  = os.path.join(DEPTH_DIR, image_id + ".npy")
    bbox_path   = os.path.join(BBOX_ORIENT_DIR, image_id + "_bboxes.npy")
    orient_path = os.path.join(BBOX_ORIENT_DIR, image_id + "_orient.npy")

    # Required files check
    for p in [depth_path, bbox_path, orient_path, FOCAL_TXT]:
        if not os.path.isfile(p):
            return None, f"missing:{p}"

    img = cv2.imread(img_path)
    if img is None:
        return None, "missing-image"

    H, W = img.shape[:2]

    try:
        f_px = load_fpx_from_txt(FOCAL_TXT, image_id)
    except Exception as e:
        return None, f"missing-fpx:{e}"

    try:
        res = identify_start_end_bboxes(
            image_path=img_path,
            depth_npy_path=depth_path,
            bboxes_npy_path=bbox_path,
            orient_npy_path=orient_path,
            f_px=f_px,
        )
    except Exception as e:
        return None, f"fail-identify:{e}"

    start_bbox = res["start_bbox_xyxy"]
    end_bbox   = res["end_bbox_xyxy"]

    end_visible, end_loc, end_turn = endpoint_fields(end_bbox, W)
    start_visible, start_loc, start_turn = endpoint_fields(start_bbox, W)

    pred = {
        "image_id": image_id,
        "image_path": os.path.abspath(img_path),

        "end_of_line_visible": end_visible,
        "end_of_line_location_if_visible": end_loc,
        "direction_to_turn_to_see_end_if_not_visible": end_turn,

        "start_of_line_visible": start_visible,
        "start_of_line_location_if_visible": start_loc,
        "direction_to_turn_to_see_start_if_not_visible": start_turn,
    }

    return pred, "ok"


# ---------- MAIN ----------
def main():

    # img_path="$LINEFINDER_ROOT/Images/QueuesOutdoors/outdoor_003.jpg"
    # pred, status = process_one_image(img_path)
    # print(pred)
    # exit()
    
    image_paths = read_image_list(IMAGE_LIST_TXT)

    print(f"Loaded {len(image_paths)} image paths")

    rows = []
    failures = 0

    for img_path in image_paths:
        pred, status = process_one_image(img_path)

        if status != "ok":
            failures += 1
            rows.append({
                "image_id": os.path.splitext(os.path.basename(img_path))[0],
                "image_path": img_path,
                "status": status
            })
            continue

        pred["status"] = "ok"
        rows.append(pred)

    cols = [
        "image_id",
        "image_path",
        "status",
        "end_of_line_visible",
        "end_of_line_location_if_visible",
        "direction_to_turn_to_see_end_if_not_visible",
        "start_of_line_visible",
        "start_of_line_location_if_visible",
        "direction_to_turn_to_see_start_if_not_visible",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            for c in cols:
                r.setdefault(c, "")
            writer.writerow({c: r[c] for c in cols})

    print(f"\nSaved predictions to {OUTPUT_CSV}")
    print(f"Failures: {failures}/{len(image_paths)}")


if __name__ == "__main__":
    main()