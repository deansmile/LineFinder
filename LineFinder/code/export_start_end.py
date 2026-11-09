"""Write the start/end person selection for every image to start_end_selection.csv.

Uses identify_queue_start_end.identify_start_end_bboxes, the same function that
batch_line_visibility_nogt.py and the description scripts use. Inputs come from
$LINEFINDER_ROOT (depth_map/, bbox_orient/, focal_length_px.txt, image_paths.txt).
"""
import argparse
import csv
import os

from lf_paths import lf_path
from identify_queue_start_end import identify_start_end_bboxes, load_fpx_from_txt

HEADER = ["image_id", "image_path", "alignment_score", "start_orig_idx", "end_orig_idx",
          "start_bbox_xyxy", "end_bbox_xyxy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=lf_path("image_paths.txt"))
    ap.add_argument("--out", default="start_end_selection.csv")
    args = ap.parse_args()

    with open(args.list, encoding="utf-8") as f:
        paths = [l.strip() for l in f if l.strip()]
    rows = []
    for p in paths:
        image_id = os.path.splitext(os.path.basename(p))[0]
        try:
            res = identify_start_end_bboxes(
                image_path=p,
                depth_npy_path=lf_path("depth_map", image_id + ".npy"),
                bboxes_npy_path=lf_path("bbox_orient", image_id + "_bboxes.npy"),
                orient_npy_path=lf_path("bbox_orient", image_id + "_orient.npy"),
                f_px=load_fpx_from_txt(lf_path("focal_length_px.txt"), image_id),
            )
            rows.append([image_id, p, "%.6f" % float(res["score"]), res["start_orig_idx"], res["end_orig_idx"],
                         [float(x) for x in res["start_bbox_xyxy"].tolist()],
                         [float(x) for x in res["end_bbox_xyxy"].tolist()]])
        except Exception as e:
            print("[WARN]", image_id, e)
            rows.append([image_id, p, "", "", "", "", ""])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print("wrote", len(rows), "rows to", args.out)


if __name__ == "__main__":
    main()
