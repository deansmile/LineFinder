"""Timed LineFinder logic pass (RANSAC line fit + endpoint logic), `annot` environment.

Calls the same per-image functions the batch scripts use, on the depth maps and
boxes in $LINEFINDER_ROOT:
  - count:      batch_number_of_people.count_people_for_image   (RANSAC inliers)
  - visibility: batch_line_visibility_nogt.process_one_image    (start/end selection
                via identify_queue_start_end.identify_start_end_bboxes + edge rule)
  - facing:     batch_queue_direction.classify_queue_for_image
Each step fits its own RANSAC line, as in the pipeline. CPU only.

Writes the same CSV formats as the batch scripts, plus start_end_selection.csv,
into --out-dir, and per-image timings to --timing-csv.
"""
import argparse
import csv
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Timer, TimingLog, image_size, read_list, write_env  # noqa: E402

import batch_number_of_people as bnp  # noqa: E402
import batch_line_visibility_nogt as blv  # noqa: E402
import batch_queue_direction as bqd  # noqa: E402

# Capture the start/end selection that process_one_image computes (no extra work).
_last_start_end = {}
_orig_identify = blv.identify_start_end_bboxes


def _capturing_identify(*a, **kw):
    res = _orig_identify(*a, **kw)
    _last_start_end.clear()
    _last_start_end.update(res)
    return res


blv.identify_start_end_bboxes = _capturing_identify

VIS_COLS = ["image_id", "image_path", "status",
            "end_of_line_visible", "end_of_line_location_if_visible", "direction_to_turn_to_see_end_if_not_visible",
            "start_of_line_visible", "start_of_line_location_if_visible", "direction_to_turn_to_see_start_if_not_visible"]


def run_one(p, focal):
    image_id = os.path.splitext(os.path.basename(p))[0]
    out, t = {}, {}
    with Timer() as tm:
        if image_id in focal:
            out["count"] = bnp.count_people_for_image(image_path=p, depth_dir=bnp.DEPTH_DIR,
                                                      bbox_orient_dir=bnp.BBOX_ORIENT_DIR, f_px=focal[image_id])
        else:
            out["count"] = (None, "missing-focal")
    t["count"] = tm.seconds
    _last_start_end.clear()
    with Timer() as tm:
        out["vis"] = blv.process_one_image(p)
    t["visibility"] = tm.seconds
    out["start_end"] = dict(_last_start_end)
    with Timer() as tm:
        out["facing"] = bqd.classify_queue_for_image(image_path=p, depth_dir=bqd.DEPTH_DIR,
                                                     bbox_orient_dir=bqd.BBOX_ORIENT_DIR,
                                                     f_px=focal[image_id]) if image_id in focal else None
    t["facing"] = tm.seconds
    return image_id, out, t


def start_end_row(image_id, p, res):
    if not res:
        return [image_id, p, "", "", "", "", ""]
    return [image_id, p, "%.6f" % float(res["score"]), res["start_orig_idx"], res["end_orig_idx"],
            [float(x) for x in res["start_bbox_xyxy"].tolist()], [float(x) for x in res["end_bbox_xyxy"].tolist()]]


START_END_HEADER = ["image_id", "image_path", "alignment_score", "start_orig_idx", "end_orig_idx",
                    "start_bbox_xyxy", "end_bbox_xyxy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timing-csv", required=True)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    write_env(args.env_dir, "linefinder_logic")
    os.makedirs(args.out_dir, exist_ok=True)
    paths = read_list(args.list)
    focal = bnp.load_focal_lengths(bnp.FOCAL_TXT)

    for p in paths[:args.warmup]:
        run_one(p, focal)

    log = TimingLog(args.timing_csv)
    count_rows, vis_rows, facing_rows, se_rows = [], [], [], []
    for i, p in enumerate(paths):
        w, h = image_size(p)
        image_id, out, t = run_one(p, focal)
        note = "separate CPU pass; includes loading depth/box .npy"
        log.add("LineFinder", "RANSAC count", image_id, w, h, t["count"], note)
        log.add("LineFinder", "RANSAC start/end + visibility", image_id, w, h, t["visibility"], note)
        log.add("LineFinder", "RANSAC facing", image_id, w, h, t["facing"], note)
        log.add("LineFinder", "RANSAC + endpoint logic (total)", image_id, w, h, sum(t.values()), note)

        count, status = out["count"]
        count_rows.append([p, image_id, "" if count is None else count, status])
        pred, vstatus = out["vis"]
        if vstatus != "ok":
            pred = {"image_id": image_id, "image_path": p, "status": vstatus}
        else:
            pred["status"] = "ok"
        vis_rows.append({c: pred.get(c, "") for c in VIS_COLS})
        if out["facing"] is not None:
            facing_rows.append([p, out["facing"]])
        se_rows.append(start_end_row(image_id, p, out["start_end"]))
        print("[%d/%d] %s count=%s vis=%s facing=%s %.3fs" % (i + 1, len(paths), image_id, count, vstatus,
                                                              out["facing"], sum(t.values())))
    log.close()

    with open(os.path.join(args.out_dir, "OLD_number_of_people.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_path", "image_id", "OL_number_of_people", "status"]); w.writerows(count_rows)
    with open(os.path.join(args.out_dir, "line_visibility_new.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VIS_COLS); w.writeheader(); w.writerows(vis_rows)
    with open(os.path.join(args.out_dir, "OL_queue_facing_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_path", "facing_direction"]); w.writerows(facing_rows)
    with open(os.path.join(args.out_dir, "start_end_selection.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(START_END_HEADER); w.writerows(se_rows)


if __name__ == "__main__":
    main()
