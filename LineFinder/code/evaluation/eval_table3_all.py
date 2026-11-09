"""Table 3 (people counting) for all four methods on one GT, plus per-image errors.

GT counts and VLM predictions are read with the benchmark evaluators' own loaders and
parse_int_value (eval_number_of_people_{gpt,gemini}.py for Luna/Olivia,
eval_number_of_people_{gpt,gemini,kimi}_old.py for the older images). Kimi's Luna/Olivia
file uses the GPT question keys, as in the benchmark. LineFinder: a count CSV
(image_id, OL_number_of_people) from batch_number_of_people.py.
Images marked too hard are skipped; an image without a usable prediction is excluded and
listed. Metrics as in the benchmark (relative error skips GT = 0 unless pred = 0).

Usage: python eval_table3_all.py --gt-root <root> --lf-counts <csv> --bbox-dir <dir> --out-dir <dir>
"""
import argparse
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import eval_number_of_people_gpt as gpt_lo          # noqa: E402
import eval_number_of_people_gemini as gemini_lo    # noqa: E402
import eval_number_of_people_gpt_old as gpt_old     # noqa: E402
import eval_number_of_people_gemini_old as gemini_old  # noqa: E402
import eval_number_of_people_kimi_old as kimi_old   # noqa: E402
from eval_table1_all import load_gt                 # noqa: E402

parse = gpt_lo.parse_int_value


def gt_count(group, obj):
    if group == "lo":
        return parse(obj["annotation"].get(gpt_lo.QUESTION))
    return parse(obj.get("people", {}).get("number_of_people"))


def vlm_counts(lo_loader, lo_file, lo_key, old_loader, old_file, old_key):
    out = {}
    for loader, f, key in [(lo_loader, lo_file, lo_key), (old_loader, old_file, old_key)]:
        for k, pred in loader(f).items():
            out[os.path.splitext(os.path.basename(k))[0]] = parse(pred.get(key)) if isinstance(pred, dict) else None
    return out


def metrics(pairs):
    e = [p - g for g, p in pairs]
    n = len(e)
    rel = [abs(p - g) / g if g != 0 else 0.0 for g, p in pairs if g != 0 or p == 0]
    return {"n": n, "MAE": sum(map(abs, e)) / n, "RMSE": math.sqrt(sum(x * x for x in e) / n),
            "mean_signed_error": sum(e) / n, "mean_relative_error": sum(rel) / len(rel),
            "exact": sum(x == 0 for x in e) / n, "within_1": sum(abs(x) <= 1 for x in e) / n,
            "within_2": sum(abs(x) <= 2 for x in e) / n,
            "undercount": sum(x < 0 for x in e) / n, "overcount": sum(x > 0 for x in e) / n}


def n_boxes(path):
    if not os.path.exists(path):
        return None
    m = re.search(rb"'shape': \((\d+)", open(path, "rb").read(256))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--pred-dir", default=os.path.dirname(HERE))
    ap.add_argument("--lf-counts", required=True)
    ap.add_argument("--bbox-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    P = lambda f: os.path.join(args.pred_dir, f)
    gt = {k: (g, gt_count(g, o)) for k, (g, o) in load_gt(args.gt_root).items()}

    preds = {
        "GPT": vlm_counts(gpt_lo.load_gpt_predictions, P("gpt_line_luna_olivia.jsonl"), gpt_lo.QUESTION,
                          gpt_old.load_gpt_predictions, P("gpt_line_old.jsonl"), gpt_old.GPT_COUNT_KEY),
        "Gemini": vlm_counts(gemini_lo.load_gemini_predictions, P("gemini_line_luna_olivia.jsonl"), gemini_lo.QUESTION_GEMINI,
                             gemini_old.load_gemini_predictions, P("gemini_line_old.jsonl"), gemini_old.GEMINI_COUNT_KEY),
        "Kimi": vlm_counts(gpt_lo.load_gpt_predictions, P("kimi_line_luna_olivia.jsonl"), gpt_lo.QUESTION,
                           kimi_old.load_kimi_predictions, P("kimi_line_old.jsonl"), kimi_old.KIMI_COUNT_KEY),
        "LineFinder": {r["image_id"]: parse(r["OL_number_of_people"]) for r in csv.DictReader(open(args.lf_counts))},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    res = {"n_gt_images": len(gt)}
    for m, pr in preds.items():
        pairs, missing = [], []
        for iid, (g, c) in sorted(gt.items()):
            if c is None or pr.get(iid) is None:
                missing.append(iid)
                continue
            pairs.append((c, pr[iid]))
        res[m] = metrics(pairs)
        res[m]["excluded"] = missing
    # per-image LineFinder detail (step 4)
    with open(os.path.join(args.out_dir, "linefinder_count_errors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group", "gt", "detected_boxes", "ransac_inliers", "error"])
        for iid, (g, c) in sorted(gt.items()):
            p = preds["LineFinder"].get(iid)
            w.writerow([iid, g, c, n_boxes(os.path.join(args.bbox_dir, iid + "_bboxes.npy")), p,
                        "" if p is None or c is None else p - c])
    json.dump(res, open(os.path.join(args.out_dir, "table3_427.json"), "w"), indent=1)
    for m in preds:
        r = res[m]
        print(f"{m:10s} n={r['n']} " + " ".join(f"{k}={r[k]:.3f}" for k in
              ["MAE", "RMSE", "mean_signed_error", "mean_relative_error", "exact", "within_1", "within_2",
               "undercount", "overcount"]) + f" excluded={r['excluded']}")


if __name__ == "__main__":
    main()
