"""Table 1 (line facing, visibility, location, movement) for all four methods on one GT.

Pools the two image groups the benchmark scores separately:
  - Luna/Olivia (408 images, `annotation` schema): VLMs scored with the
    eval_<m>_luna_olivia.py rules; GT read from the JSONs.
  - older images (109, nested schema): VLMs scored with eval_gpt.py (GPT-4o run),
    eval_gemini_old.py, eval_kimi_old.py rules.
  - LineFinder: line_visibility_new.csv + facing CSVs vs GT_json_canonical.csv
    (eval_visibility_our.py rules), facing from the queue_facing CSVs.
Images marked too hard are skipped. A missing prediction is excluded (as in the
benchmark) and reported.

Facing is reported two ways:
  merged   : sideways-left / sideways-right / sideways all count as "sideways"
  unmerged : exact label (left/right must match; older-format GT has only "sideways")
"overall" uses the merged facing value so all methods are scored the same way;
overall_unmerged is also reported.

Usage: python eval_table1_all.py --gt-root <LINEFINDER_ROOT with GT_json/ and GT_json_canonical.csv>
         --lf-vis <line_visibility_new.csv> --lf-facing <facing csv> [<facing csv> ...]
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import eval_gpt_luna_olivia as gpt_lo          # noqa: E402
import eval_gemini_luna_olivia as gemini_lo    # noqa: E402
import eval_kimi_luna_olivia as kimi_lo        # noqa: E402
import eval_gpt as gpt_old                     # noqa: E402  (GPT-4o run on the older images)
import eval_gemini_old as gemini_old           # noqa: E402
import eval_kimi_old as kimi_old               # noqa: E402

FIELDS = ["facing", "end_visible", "end_location", "end_move", "start_visible", "start_location", "start_move"]
LO_FIELDS = dict(zip(FIELDS, gpt_lo.EVAL_FIELDS))
OLD_FIELDS = dict(zip(FIELDS, gemini_old.EVAL_FIELDS))
CANON_FIELDS = dict(zip(FIELDS, ["line facing direction", "end visible", "end location", "end direction to turn",
                                 "start visible", "start location", "start direction to turn"]))
LF_VIS_FIELDS = {"end_visible": "end_of_line_visible", "end_location": "end_of_line_location_if_visible",
                 "end_move": "direction_to_turn_to_see_end_if_not_visible", "start_visible": "start_of_line_visible",
                 "start_location": "start_of_line_location_if_visible",
                 "start_move": "direction_to_turn_to_see_start_if_not_visible"}


def facing_label(x, merge):
    s = "" if x is None else str(x).strip().lower()
    if s.startswith("facing "):
        s = s[len("facing "):]
    if merge and s in {"sideways-left", "sideways-right", "sideways"}:
        return "sideways"
    return s


def gt_key(k):
    return k[:-4] if k.lower().endswith(".jpg") else k   # IMG_6728.jpg.json -> IMG_6728


def load_gt(gt_root):
    """Returns {image_id: (group, gt_obj)} for not-too-hard images."""
    out = {}
    for dirpath, _, files in os.walk(os.path.join(gt_root, "GT_json")):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            obj = json.load(open(os.path.join(dirpath, fn), encoding="utf-8"))
            if "summary" in obj:
                continue
            key = gt_key(os.path.splitext(fn)[0])
            if "annotation" in obj:
                if gpt_lo.is_too_hard(obj):
                    continue
                out[key] = ("lo", obj)
            else:
                out[key] = ("old", obj)
    return out


def score_vlm(name, lo_mod, lo_file, old_mod, old_file, gt):
    """Per-image correctness dict {image_id: {field: bool, facing_unmerged: bool}}."""
    per = {}
    lo_res = lo_mod.read_result_file(lo_file)
    old_res = old_mod.read_result_file(old_file)
    for iid, (group, obj) in gt.items():
        if group == "lo":
            if iid not in lo_res:
                continue
            pred, ann = lo_res[iid], gpt_lo.extract_gt_annotation(obj)
            row = {}
            for f in FIELDS:  # benchmark rule: normalize_text equality
                row[f] = lo_mod.normalize_text(pred.get(LO_FIELDS[f], "N/A")) == lo_mod.normalize_text(ann.get(LO_FIELDS[f], "N/A"))
            pv, gv = pred.get(LO_FIELDS["facing"]), ann.get(LO_FIELDS["facing"])
        else:
            if iid not in old_res:
                continue
            pred = old_res[iid]
            flat = old_mod.normalize_gt_to_flat(obj)
            row = {}
            for f in FIELDS:
                k = OLD_FIELDS[f]
                if old_mod is gpt_old:  # eval_gpt.py rule: str().strip().lower() equality
                    row[f] = str(pred.get(k)).strip().lower() == str(flat.get(k)).strip().lower()
                elif f == "facing":
                    row[f] = old_mod.normalize_line_facing_direction(pred.get(k, "N/A")) == \
                        old_mod.normalize_line_facing_direction(flat.get(k, "N/A"))
                else:
                    row[f] = old_mod.normalize_text(pred.get(k, "N/A")) == old_mod.normalize_text(flat.get(k, "N/A"))
            pv, gv = pred.get(OLD_FIELDS["facing"]), flat.get(OLD_FIELDS["facing"])
        row["facing_benchmark"] = row["facing"]
        row["facing_merged"] = facing_label(pv, True) == facing_label(gv, True)
        row["facing_unmerged"] = facing_label(pv, False) == facing_label(gv, False)
        row["group"] = group
        per[iid] = row
    return per


def score_linefinder(gt, gt_root, vis_csv, facing_csvs):
    canon = {gt_key(r["image_id"]): r for r in csv.DictReader(open(os.path.join(gt_root, "GT_json_canonical.csv")))}
    vis = {r["image_id"]: r for r in csv.DictReader(open(vis_csv)) if r.get("status", "ok") == "ok"}
    fac = {}
    for p in facing_csvs:
        for r in csv.DictReader(open(p)):
            fac[os.path.splitext(os.path.basename(r["image_path"]))[0]] = r["facing_direction"]
    per = {}
    for iid, (group, obj) in gt.items():
        if iid not in vis or iid not in fac or iid not in canon:
            continue
        g = canon[iid]
        row = {}
        for f, col in LF_VIS_FIELDS.items():  # eval_visibility_our.py rule
            row[f] = norm_vis(f, vis[iid][col]) == norm_vis(f, g[CANON_FIELDS[f]])
        raw_gt = gpt_lo.extract_gt_annotation(obj).get(LO_FIELDS["facing"]) if group == "lo" else g[CANON_FIELDS["facing"]]
        row["facing_merged"] = facing_label(fac[iid], True) == facing_label(raw_gt, True)
        row["facing_unmerged"] = facing_label(fac[iid], False) == facing_label(raw_gt, False)
        row["facing"] = row["facing_benchmark"] = row["facing_merged"]  # LineFinder was always scored merged
        row["group"] = group
        per[iid] = row
    return per


def norm_vis(field, x):
    s = "n/a" if x is None or str(x).strip() == "" else str(x).strip().lower()
    if s in {"n/a", "na", "none", "null", "unknown"}:
        s = "n/a"
    if field.endswith("visible"):
        s = {"true": "yes", "1": "yes", "false": "no", "0": "no"}.get(s, s)
    return s


def summarize(per, n_gt):
    n = len(per)
    out = {"n_evaluated": n, "n_missing": n_gt - n}
    for f in FIELDS[1:]:
        out[f] = 100.0 * sum(r[f] for r in per.values()) / n
    for f in ["facing_benchmark", "facing_merged", "facing_unmerged"]:
        out[f] = 100.0 * sum(r[f] for r in per.values()) / n
    six = sum(sum(r[f] for f in FIELDS[1:]) for r in per.values())
    out["overall"] = 100.0 * (six + sum(r["facing_merged"] for r in per.values())) / (7 * n)
    out["overall_unmerged"] = 100.0 * (six + sum(r["facing_unmerged"] for r in per.values())) / (7 * n)
    out["overall_benchmark_facing"] = 100.0 * (six + sum(r["facing_benchmark"] for r in per.values())) / (7 * n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--pred-dir", default=os.path.dirname(HERE))
    ap.add_argument("--lf-vis", required=True)
    ap.add_argument("--lf-facing", nargs="+", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    P = lambda f: os.path.join(args.pred_dir, f)
    gt = load_gt(args.gt_root)
    per = {
        "LineFinder": score_linefinder(gt, args.gt_root, args.lf_vis, args.lf_facing),
        "GPT": score_vlm("GPT", gpt_lo, P("gpt_line_luna_olivia.jsonl"), gpt_old, P("gpt_line_old.jsonl"), gt),
        "Gemini": score_vlm("Gemini", gemini_lo, P("gemini_line_luna_olivia.jsonl"), gemini_old, P("gemini_line_old.jsonl"), gt),
        "Kimi": score_vlm("Kimi", kimi_lo, P("kimi_line_luna_olivia.jsonl"), kimi_old, P("kimi_line_old.jsonl"), gt),
    }
    res = {"n_gt_images": len(gt)}
    for m, p in per.items():
        res[m] = {"all": summarize(p, len(gt))}
        for grp in ["lo", "old"]:
            sub = {k: v for k, v in p.items() if v["group"] == grp}
            res[m][grp] = summarize(sub, sum(1 for g, _ in gt.values() if g == grp))
        res[m]["missing_ids"] = sorted(set(gt) - set(p))
    print(json.dumps(res, indent=1))
    if args.json_out:
        json.dump(res, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
