from lf_paths import lf_path  # LINEFINDER_ROOT env var
# batch_line_visibility.py
import os
import csv
import json
import numpy as np
import cv2

from eval_gpt import load_all_json_recursive_with_paths  # same as your attached script
from identify_queue_start_end import identify_start_end_bboxes, load_fpx_from_txt


GT_FOLDER = lf_path("GT_json")
IMG_FOLDER = lf_path("Images")

DEPTH_DIR = lf_path("depth_map")
BBOX_ORIENT_DIR = lf_path("bbox_orient")
FOCAL_TXT = lf_path("focal_length_px.txt")

def flatten_and_fix_gt(gt_entry: dict) -> dict:
    """
    Input GT format (nested):
      gt_entry["end_of_line"]["visible"]
      gt_entry["end_of_line"]["location_if_visible"]
      gt_entry["end_of_line"]["direction_to_turn_if_not_visible"]
    Output (flat, matching prediction keys):
      end_of_line_visible
      end_of_line_location_if_visible
      direction_to_turn_to_see_end_if_not_visible
    Same for start.

    Also enforces:
      visible=="yes" => turn="N/A"
      visible=="no"  => location="N/A"
    """
    def get_nested(side: str, key: str, default=""):
        obj = gt_entry.get(f"{side}_of_line", {})
        if not isinstance(obj, dict):
            return default
        return obj.get(key, default)

    flat = {
        "end_of_line_visible": str(get_nested("end", "visible", "")).strip().lower(),
        "end_of_line_location_if_visible": str(get_nested("end", "location_if_visible", "N/A")).strip().lower(),
        "direction_to_turn_to_see_end_if_not_visible": str(get_nested("end", "direction_to_turn_if_not_visible", "N/A")).strip().lower(),

        "start_of_line_visible": str(get_nested("start", "visible", "")).strip().lower(),
        "start_of_line_location_if_visible": str(get_nested("start", "location_if_visible", "N/A")).strip().lower(),
        "direction_to_turn_to_see_start_if_not_visible": str(get_nested("start", "direction_to_turn_if_not_visible", "N/A")).strip().lower(),
    }

    # Canonicalize / repair consistency
    def fix(prefix: str):
        vis_k = f"{prefix}_of_line_visible"
        loc_k = f"{prefix}_of_line_location_if_visible"
        turn_k = f"direction_to_turn_to_see_{prefix}_if_not_visible"

        vis = flat.get(vis_k, "")
        if vis not in ("yes", "no"):
            return

        if vis == "yes":
            flat[turn_k] = "N/A"
            valid_locs = {"far left","center left","center","center right","far right"}
            if flat.get(loc_k, "N/A") not in valid_locs:
                flat[loc_k] = "N/A"
        else:  # vis == "no"
            flat[loc_k] = "N/A"
            valid_turns = {"left","right"}
            if flat.get(turn_k, "N/A") not in valid_turns:
                flat[turn_k] = "N/A"

    fix("end")
    fix("start")

    # Store canonical casing (match your prediction strings)
    # visible: yes/no already lowercase
    # location: lowercase; N/A uppercase
    for k in ["end_of_line_location_if_visible", "start_of_line_location_if_visible",
              "direction_to_turn_to_see_end_if_not_visible", "direction_to_turn_to_see_start_if_not_visible"]:
        v = flat.get(k, "N/A")
        flat[k] = "N/A" if v in ("n/a", "na", "") else v

    for k in ["end_of_line_visible", "start_of_line_visible"]:
        v = flat.get(k, "")
        flat[k] = "yes" if v == "yes" else ("no" if v == "no" else "")

    return flat

def normalize_visibility_fields(gt: dict) -> dict:
    """
    Fix inconsistent GT fields in-place according to the visibility rules.
    Returns a new dict (copy) with repaired fields.
    """
    gt = dict(gt)  # shallow copy

    def norm_side(prefix: str):
        # prefix in {"start", "end"}
        vis_k = f"{prefix}_of_line_visible"
        loc_k = f"{prefix}_of_line_location_if_visible"
        turn_k = f"direction_to_turn_to_see_{prefix}_if_not_visible"

        vis = str(gt.get(vis_k, "")).strip().lower()
        if vis not in ("yes", "no"):
            return  # leave as-is if missing/invalid

        # normalize to canonical case
        gt[vis_k] = "yes" if vis == "yes" else "no"

        if gt[vis_k] == "yes":
            # visible => turn must be N/A
            gt[turn_k] = "N/A"

            # location can stay if valid, otherwise N/A
            valid_locs = {"far left", "center left", "center", "center right", "far right"}
            loc = str(gt.get(loc_k, "N/A")).strip().lower()
            if loc in valid_locs:
                # store canonical case exactly
                gt[loc_k] = loc
            else:
                gt[loc_k] = "N/A"

        else:
            # not visible => location must be N/A
            gt[loc_k] = "N/A"

            # turn can stay if valid; otherwise N/A
            valid_turn = {"left", "right"}
            turn = str(gt.get(turn_k, "N/A")).strip().lower()
            if turn in valid_turn:
                gt[turn_k] = turn
            else:
                gt[turn_k] = "N/A"

    norm_side("end")
    norm_side("start")
    return gt

def get_images_with_gt(img_folder, gt_keys):
    """Same logic as in batch_queue_direction.py: match by basename (no extension)."""
    matched = []
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    for root, _, files in os.walk(img_folder):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                key = os.path.splitext(fname)[0]
                if key in gt_keys:
                    matched.append(os.path.join(root, fname))
    return matched


def _bbox_edge_flags(bbox_xyxy, W, H, margin_px):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    near_left   = x1 <= margin_px
    near_right  = x2 >= (W - 1 - margin_px)
    near_top    = y1 <= margin_px
    near_bottom = y2 >= (H - 1 - margin_px)
    touches_any = near_left or near_right or near_top or near_bottom
    return touches_any, near_left, near_right, near_top, near_bottom


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


def endpoint_fields(bbox_xyxy, W, H, margin_px):
    """
    Implements your rule:
      - if bbox touches/is close to any edge -> not visible
      - if not visible: turn left if near left edge else right
      - if visible: location bucket by bbox center x
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    cx = 0.5 * (x1 + x2)

    touches_any, near_left, near_right, near_top, near_bottom = _bbox_edge_flags(
        bbox_xyxy, W, H, margin_px
    )

    if touches_any:
        visible = "no"
        location = "N/A"
        turn = "left" if near_left else "right"
    else:
        visible = "yes"
        location = _location_bucket_from_center_x(cx, W)
        turn = "N/A"

    return visible, location, turn


def process_one_image(img_path, gt_entry, margin_px=10):
    image_id = os.path.splitext(os.path.basename(img_path))[0]

    # Paths
    depth_path = os.path.join(DEPTH_DIR, image_id + ".npy")
    bbox_path  = os.path.join(BBOX_ORIENT_DIR, image_id + "_bboxes.npy")
    orient_path = os.path.join(BBOX_ORIENT_DIR, image_id + "_orient.npy")

    # Required files check
    for p in [depth_path, bbox_path, orient_path, FOCAL_TXT]:
        if not os.path.isfile(p):
            return None, f"missing:{p}"

    # Read image for W,H
    img = cv2.imread(img_path)
    if img is None:
        return None, "missing-image"
    H, W = img.shape[:2]

    # Look up focal length
    try:
        f_px = load_fpx_from_txt(FOCAL_TXT, image_id)
    except Exception as e:
        return None, f"missing-fpx:{e}"

    # Identify start/end bboxes
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

    start_bbox = res["start_bbox_xyxy"]  # START = head
    end_bbox   = res["end_bbox_xyxy"]    # END = tail

    # Compute fields
    end_visible, end_loc, end_turn = endpoint_fields(end_bbox, W, H, margin_px)
    start_visible, start_loc, start_turn = endpoint_fields(start_bbox, W, H, margin_px)

    pred = {
        "image_id": image_id,
        "image_path": img_path,

        "end_of_line_visible": end_visible,
        "end_of_line_location_if_visible": end_loc,
        "direction_to_turn_to_see_end_if_not_visible": end_turn,

        "start_of_line_visible": start_visible,
        "start_of_line_location_if_visible": start_loc,
        "direction_to_turn_to_see_start_if_not_visible": start_turn,
    }

    # Pull GT fields if present
    gt = {}
    if isinstance(gt_entry, dict):
        for k in [
            "end_of_line_visible",
            "end_of_line_location_if_visible",
            "direction_to_turn_to_see_end_if_not_visible",
            "start_of_line_visible",
            "start_of_line_location_if_visible",
            "direction_to_turn_to_see_start_if_not_visible",
        ]:
            if k in gt_entry:
                gt[k] = gt_entry[k]

    return (pred, gt), "ok"


def main():
    # Load GT jsons (same as batch_queue_direction.py)
    gt_dict, gt_paths = load_all_json_recursive_with_paths(GT_FOLDER)
    gt_keys = set(gt_dict.keys())
    print(f"Loaded {len(gt_keys)} GT JSONs.")

    # Find matching images (same logic)
    image_paths = get_images_with_gt(IMG_FOLDER, gt_keys)
    print(f"Found {len(image_paths)} images that have GT JSONs.")

    margin_px = 10  # tweak if needed

    # Collect results
    rows = []
    correct = {k: 0 for k in [
        "end_of_line_visible",
        "end_of_line_location_if_visible",
        "direction_to_turn_to_see_end_if_not_visible",
        "start_of_line_visible",
        "start_of_line_location_if_visible",
        "direction_to_turn_to_see_start_if_not_visible",
    ]}
    total = {k: 0 for k in correct.keys()}

    failures = 0

    for img_path in image_paths:
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        # 1) Build FLAT + FIXED GT (nested -> flat, and enforce N/A rules)
        gt_flat = flatten_and_fix_gt(gt_dict.get(image_id, {}))

        # 2) Run prediction; pass gt_flat in (optional) for logging
        out, status = process_one_image(img_path, gt_flat, margin_px=margin_px)
        if status != "ok":
            failures += 1
            rows.append({
                "image_id": image_id,
                "image_path": img_path,
                "status": status,
            })
            print(f"[WARN] {image_id}: {status}")
            continue

        pred, gt = out  # gt should be the flat dict (or subset) returned by process_one_image

        # add GT columns + compute accuracy
        row = dict(pred)
        row["status"] = "ok"

        # write GT columns (only non-empty)
        for k, v in gt.items():
            if v != "" and v is not None:
                row[f"gt_{k}"] = v

        # score
        for k in correct.keys():
            if gt.get(k, "") != "":   # field exists in GT
                total[k] += 1
                if str(pred[k]).strip().lower() == str(gt[k]).strip().lower():
                    correct[k] += 1

        rows.append(row)
        # Print only incorrect visibility cases
        end_vis_wrong = (
            gt.get("end_of_line_visible", "") != "" and
            pred["end_of_line_visible"] != gt["end_of_line_visible"]
        )

        start_vis_wrong = (
            gt.get("start_of_line_visible", "") != "" and
            pred["start_of_line_visible"] != gt["start_of_line_visible"]
        )

        if end_vis_wrong or start_vis_wrong:
            end_loc = pred["end_of_line_location_if_visible"] if pred["end_of_line_visible"] == "yes" else "N/A"
            gt_end_loc = gt.get("end_of_line_location_if_visible", "N/A")

            start_loc = pred["start_of_line_location_if_visible"] if pred["start_of_line_visible"] == "yes" else "N/A"
            gt_start_loc = gt.get("start_of_line_location_if_visible", "N/A")

            print(
                f"[VIS ERROR] {image_id} | "
                f"end: pred={pred['end_of_line_visible']} "
                f"(loc={end_loc}) gt={gt.get('end_of_line_visible')} "
                f"start: pred={pred['start_of_line_visible']} "
                f"(loc={start_loc}) gt={gt.get('start_of_line_visible')} "
            )

    # Print accuracies
    print("\n=== Accuracy (only where GT field exists) ===")
    for k in correct.keys():
        if total[k] == 0:
            print(f"{k}: N/A (no GT)")
        else:
            acc = correct[k] / total[k]
            print(f"{k}: {acc:.4f} ({correct[k]}/{total[k]})")

    print(f"\nFailures: {failures}/{len(image_paths)}")

    # Save CSV
    out_csv = "line_visibility_results.csv"
    # gather all possible columns (pred + gt + status)
    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    all_cols = sorted(all_cols)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
