#!/usr/bin/env python3
from lf_paths import lf_path  # LINEFINDER_ROOT env var
import csv
import os

GT_CSV = lf_path("GT_json_canonical.csv")
PRED_CSV = lf_path("code/line_visibility_new.csv")
OUT_TXT = lf_path("fully_correct_image_names.txt")


def norm(x):
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"na", "n/a", "none", ""}:
        return "N/A"
    return x.lower()


def load_gt(gt_csv):
    gt_dict = {}
    with open(gt_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"].strip()

            # Skip too-hard images
            if norm(row.get("too_hard", "")) == "yes":
                continue

            gt_dict[image_id] = {
                "end visible": norm(row.get("end visible", "")),
                "end location": norm(row.get("end location", "")),
                "end direction to turn": norm(row.get("end direction to turn", "")),
                "start visible": norm(row.get("start visible", "")),
                "start location": norm(row.get("start location", "")),
                "start direction to turn": norm(row.get("start direction to turn", "")),
            }
    return gt_dict


def main():
    gt_dict = load_gt(GT_CSV)
    fully_correct = []

    with open(PRED_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"].strip()

            # only evaluate rows with usable status
            if norm(row.get("status", "")) != "ok":
                continue

            if image_id not in gt_dict:
                continue

            gt = gt_dict[image_id]

            pred = {
                "end visible": norm(row.get("end_of_line_visible", "")),
                "end location": norm(row.get("end_of_line_location_if_visible", "")),
                "end direction to turn": norm(row.get("direction_to_turn_to_see_end_if_not_visible", "")),
                "start visible": norm(row.get("start_of_line_visible", "")),
                "start location": norm(row.get("start_of_line_location_if_visible", "")),
                "start direction to turn": norm(row.get("direction_to_turn_to_see_start_if_not_visible", "")),
            }

            all_correct = (
                pred["end visible"] == gt["end visible"]
                and pred["end location"] == gt["end location"]
                and pred["end direction to turn"] == gt["end direction to turn"]
                and pred["start visible"] == gt["start visible"]
                and pred["start location"] == gt["start location"]
                and pred["start direction to turn"] == gt["start direction to turn"]
            )

            if all_correct:
                fully_correct.append(image_id)

    fully_correct = sorted(set(fully_correct))

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for image_id in fully_correct:
            f.write(image_id + "\n")

    print(f"Found {len(fully_correct)} fully correct images.")
    print(f"Saved image names to: {OUT_TXT}")


if __name__ == "__main__":
    main()