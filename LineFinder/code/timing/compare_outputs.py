"""Check that the timed passes produce the same outputs as the official pipeline run.

- DepthPro: depth maps (max abs diff) and focal lengths.
- JointBDOE: box/orientation arrays (same shape, max abs diff).
- LineFinder logic: the four CSVs must be identical (same inputs, seeded RANSAC).
Exit code 1 if anything differs beyond tolerance.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def focal(path):
    d = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            d[parts[0]] = float(parts[1])
    return d


def compare_npy(dir_a, dir_b, pattern):
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_a, pattern)))
    names_b = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_b, pattern)))
    worst, shape_mismatch = 0.0, []
    for n in names:
        if n not in names_b:
            continue
        a, b = np.load(os.path.join(dir_a, n)), np.load(os.path.join(dir_b, n))
        if a.shape != b.shape:
            shape_mismatch.append(n)
            continue
        if a.size:
            worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return {"files_a": len(names), "files_b": len(names_b), "only_in_one": sorted(set(names) ^ set(names_b)),
            "shape_mismatch": shape_mismatch, "max_abs_diff": worst}


def compare_csv(a, b, key_cols=None):
    ra = list(csv.reader(open(a, newline="")))
    rb = list(csv.reader(open(b, newline="")))
    diff = [i for i, (x, y) in enumerate(zip(ra, rb)) if x != y]
    return {"rows_a": len(ra), "rows_b": len(rb), "differing_rows": len(diff) + abs(len(ra) - len(rb)),
            "first_diffs": [(ra[i], rb[i]) for i in diff[:3]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--timed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-tol", type=float, default=1e-3)
    ap.add_argument("--box-tol", type=float, default=1e-2)
    args = ap.parse_args()
    O, T = args.official, args.timed

    res = {"depth": compare_npy(os.path.join(O, "depth_map"), os.path.join(T, "depth_map"), "*.npy")}
    fo, ft = focal(os.path.join(O, "focal_length_px.txt")), focal(os.path.join(T, "focal_length_px.txt"))
    res["focal"] = {"n_official": len(fo), "n_timed": len(ft),
                    "max_abs_diff": max([abs(fo[k] - ft[k]) for k in fo if k in ft] or [0.0])}
    res["boxes"] = compare_npy(os.path.join(O, "bbox_orient"), os.path.join(T, "bbox_orient"), "*.npy")
    for name in ["OLD_number_of_people.csv", "line_visibility_new.csv", "OL_queue_facing_results.csv",
                 "start_end_selection.csv"]:
        res[name] = compare_csv(os.path.join(O, name), os.path.join(T, "lf", name))

    ok = (res["depth"]["max_abs_diff"] <= args.depth_tol and not res["depth"]["only_in_one"]
          and not res["depth"]["shape_mismatch"] and res["focal"]["max_abs_diff"] <= args.depth_tol
          and res["boxes"]["max_abs_diff"] <= args.box_tol and not res["boxes"]["only_in_one"]
          and not res["boxes"]["shape_mismatch"]
          and all(res[n]["differing_rows"] == 0 for n in ["OLD_number_of_people.csv", "line_visibility_new.csv",
                                                          "OL_queue_facing_results.csv", "start_end_selection.csv"]))
    res["outputs_identical_within_tolerance"] = ok
    json.dump(res, open(args.out, "w"), indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
