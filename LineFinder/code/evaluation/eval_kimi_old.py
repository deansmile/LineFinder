import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
import json

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")

EVAL_FIELDS = [
    "line_facing_direction",
    "end_of_line_visible",
    "end_of_line_location_if_visible",
    "direction_to_turn_to_see_end_if_not_visible",
    "start_of_line_visible",
    "start_of_line_location_if_visible",
    "direction_to_turn_to_see_start_if_not_visible",
]

KIMI_TO_FLAT_FIELD = {
    "What direction is the line facing": "line_facing_direction",
    "Do we see the end of line?": "end_of_line_visible",
    "If yes, where is the end of the line in the image?": "end_of_line_location_if_visible",
    "If not, in what direction should you move camera to see the end of the line?": "direction_to_turn_to_see_end_if_not_visible",
    "Do we see the start of line?": "start_of_line_visible",
    "If yes, where is the start of the line in the image?": "start_of_line_location_if_visible",
    "If not, in what direction should you move camera to see the start of the line?": "direction_to_turn_to_see_start_if_not_visible",
}


def strip_image_extension(filename):
    base = os.path.basename(filename.strip())
    for ext in IMAGE_EXTS:
        if base.lower().endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]



def normalize_line_facing_direction(x):
    """
    Normalize old GT labels and Kimi labels to the same space.

    GT examples:
      - Facing away
      - Facing towards
      - Facing sideways
      - other

    Kimi examples:
      - away
      - towards
      - sideways-left
      - sideways-right
    """
    if x is None:
        return "n/a"

    s = str(x).strip().lower()
    if s in {"n/a", "na", "none", "null", ""}:
        return "n/a"

    mapping = {
        "facing away": "away",
        "away": "away",
        "facing towards": "towards",
        "towards": "towards",
        "facing sideways": "sideways",
        "sideways": "sideways",
        "sideways-left": "sideways",
        "sideways-right": "sideways",
        "other": "other",
    }
    return mapping.get(s, s)



def normalize_text(x, field=None):
    if field == "line_facing_direction":
        return normalize_line_facing_direction(x)

    if x is None:
        return "n/a"
    s = str(x).strip()
    if s.lower() in {"n/a", "na", "none", "null", ""}:
        return "n/a"
    return s.lower()



def normalize_prediction_fields(pred):
    out = {}
    for key, value in pred.items():
        if key in EVAL_FIELDS:
            out[key] = value
        elif key in KIMI_TO_FLAT_FIELD:
            out[KIMI_TO_FLAT_FIELD[key]] = value

    for field in EVAL_FIELDS:
        out.setdefault(field, "N/A")
    return out



def read_result_file(path):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.lower().endswith(IMAGE_EXTS):
            image_id = strip_image_extension(line)
            i += 1

            while i < len(lines) and not lines[i].strip():
                i += 1

            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1
                json_lines = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    json_lines.append(lines[i])
                    i += 1
                if i < len(lines) and lines[i].strip().startswith("```"):
                    i += 1
            else:
                json_lines = []
                brace_balance = 0
                started = False
                while i < len(lines):
                    cur = lines[i]
                    if not started and not cur.strip():
                        i += 1
                        continue
                    json_lines.append(cur)
                    brace_balance += cur.count("{") - cur.count("}")
                    if "{" in cur:
                        started = True
                    i += 1
                    if started and brace_balance <= 0:
                        break

            try:
                obj = json.loads("".join(json_lines))
                results[image_id] = normalize_prediction_fields(obj)
            except Exception as e:
                print(f"[warn] failed to parse {image_id}: {e}")
        else:
            i += 1
    return results



def load_all_json_recursive_with_paths(root_folder):
    gt_dict, gt_paths = {}, {}
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                key = os.path.splitext(fname)[0]
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        gt_dict[key] = json.load(f)
                        gt_paths[key] = os.path.abspath(fpath)
                except Exception as e:
                    print(f"[warn] Failed to load {fpath}: {e}")
    return gt_dict, gt_paths



def bool_to_yesno(x):
    if isinstance(x, bool):
        return "yes" if x else "no"
    s = str(x).strip().lower()
    if s in {"true", "yes", "1"}:
        return "yes"
    if s in {"false", "no", "0"}:
        return "no"
    return s



def normalize_gt_to_flat(gt):
    ls = gt.get("line_shape", {})
    ppl = gt.get("people", {})
    bnd = gt.get("boundary", {})
    sol = gt.get("start_of_line", {})
    eol = gt.get("end_of_line", {})

    flat = {
        "number_of_turns": int(ls.get("turns")) if isinstance(ls.get("turns"), (int, float, str)) and str(ls.get("turns")).isdigit() else int(ls.get("turns") or 0),
        "line_shape": ls.get("description", "unknown"),
        "line_facing_direction": ppl.get("direction_they_are_facing", "unknown"),
        "number_of_people_in_line": int(ppl.get("number_of_people") or 0),
        "line_purpose": gt.get("line_purpose", "unknown"),
        "start_person_description": ppl.get("start_person_description", "unknown"),
        "end_person_description": ppl.get("end_person_description", "unknown"),
        "counter_person_description": ppl.get("counter_person_description", "unknown"),
        "boundary_present": bool_to_yesno(bnd.get("boundary_present", "no")),
        "boundary_types": bnd.get("boundary_types", "none"),
        "end_of_line_visible": eol.get("visible", "no"),
        "end_of_line_location_if_visible": eol.get("location_if_visible", "N/A"),
        "direction_to_turn_to_see_end_if_not_visible": eol.get("direction_to_turn_if_not_visible", "N/A"),
        "start_of_line_visible": sol.get("visible", "no"),
        "start_of_line_location_if_visible": sol.get("location_if_visible", "N/A"),
        "direction_to_turn_to_see_start_if_not_visible": sol.get("direction_to_turn_if_not_visible", "N/A"),
        "line_completeness": gt.get("line_completeness", "partial"),
    }
    return flat



def evaluate_accuracy_with_mismatches(results, gt_dict, gt_paths):
    def norm_key(k):
        return os.path.splitext(k)[0]

    correct_counts = {k: 0 for k in EVAL_FIELDS}
    total_counts = {k: 0 for k in EVAL_FIELDS}
    mismatches = {k: [] for k in EVAL_FIELDS}

    common = set(map(norm_key, results.keys())) & set(gt_dict.keys())

    for img_norm in sorted(common):
        rkey = next(k for k in results.keys() if norm_key(k) == img_norm)
        pred_res = results[rkey]
        gt_res = normalize_gt_to_flat(gt_dict[img_norm])

        for k in EVAL_FIELDS:
            total_counts[k] += 1
            pv = normalize_text(pred_res.get(k, "N/A"), field=k)
            gv = normalize_text(gt_res.get(k, "N/A"), field=k)
            if pv == gv:
                correct_counts[k] += 1
            else:
                mismatches[k].append((rkey, pred_res.get(k), gt_res.get(k), gt_paths.get(img_norm, "N/A")))

    accuracies = {k: (correct_counts[k] / total_counts[k] if total_counts[k] else None) for k in EVAL_FIELDS}
    overall = sum(correct_counts.values()) / sum(total_counts.values()) if sum(total_counts.values()) else None
    return accuracies, overall, mismatches, sorted(common)



def print_specific_mismatches(mismatches):
    for f in EVAL_FIELDS:
        print(f"\nIncorrect {f}:")
        if not mismatches.get(f):
            print("  (none)")
            continue
        for img_name, pred_v, gt_v, gt_path in mismatches[f]:
            print(f"  {img_name} | pred={pred_v!r} | GT={gt_v!r} | GT JSON: {gt_path}")


if __name__ == "__main__":
    results = read_result_file("kimi_line_old.jsonl")
    print(f"Loaded {len(results)} records")

    folder = lf_path("GT_json")
    gt_dict, gt_paths = load_all_json_recursive_with_paths(folder)
    print(f"Loaded {len(gt_dict)} JSON files")

    acc, overall, mismatches, common_ids = evaluate_accuracy_with_mismatches(results, gt_dict, gt_paths)

    print(f"Matched {len(common_ids)} images between results and GT")
    print("Per-field accuracy:")
    for k, v in acc.items():
        print(f"  {k}: {v:.2%}" if v is not None else f"  {k}: N/A")
    print(f"Overall accuracy: {overall:.2%}" if overall is not None else "Overall accuracy: N/A")
    # print_specific_mismatches(mismatches)
