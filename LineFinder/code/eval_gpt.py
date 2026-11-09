import os
import json

def read_result_file(path):
    """
    Reads your result file where each record is:

    <filename>
    ```json
    { ... }
    ```

    Returns a dict mapping filename -> parsed JSON dict.
    """
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # line should be filename like IMG_5954.JPG
        if line.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            filename = line
            i += 1
            # skip ```json
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1
            json_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                json_lines.append(lines[i])
                i += 1
            # skip closing ```
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1
            try:
                obj = json.loads("".join(json_lines))
                results[filename[:-4]] = obj
            except Exception as e:
                print(f"[warn] failed to parse {filename}: {e}")
        else:
            i += 1
    return results

def load_all_json_recursive_with_paths(root_folder):
    """
    Recursively reads all .json files and returns:
      - gt_dict:  name(no ext) -> parsed JSON dict
      - gt_paths: name(no ext) -> absolute file path
    """
    gt_dict, gt_paths = {}, {}
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                key = os.path.splitext(fname)[0]  # strip extension
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
    # also accept strings like "true"/"false"
    s = str(x).strip().lower()
    if s in {"true", "yes", "1"}:
        return "yes"
    if s in {"false", "no", "0"}:
        return "no"
    return s  # fallback

FLAT_KEYS = [
    "number_of_turns",
    "line_shape",
    "line_facing_direction",
    "number_of_people_in_line",
    "line_purpose",
    "start_person_description",
    "end_person_description",
    "counter_person_description",
    "boundary_present",
    "boundary_types",
    "end_of_line_visible",
    "end_of_line_location_if_visible",
    "direction_to_turn_to_see_end_if_not_visible",
    "start_of_line_visible",
    "start_of_line_location_if_visible",
    "direction_to_turn_to_see_start_if_not_visible",
    "line_completeness",
]

def normalize_gt_to_flat(gt: dict) -> dict:
    """
    Convert your GT (nested) structure to the flat schema used by GPT results.
    Any missing fields are filled with sensible 'N/A' / 'unknown' defaults.
    """
    # safe gets
    ls = gt.get("line_shape", {})
    ppl = gt.get("people", {})
    bnd = gt.get("boundary", {})
    sol = gt.get("start_of_line", {})
    eol = gt.get("end_of_line", {})

    flat = {
        "number_of_turns":            int(ls.get("turns")) if isinstance(ls.get("turns"), (int, float, str)) and str(ls.get("turns")).isdigit() else int(ls.get("turns") or 0),
        "line_shape":                 ls.get("description", "unknown"),

        "line_facing_direction":      ppl.get("direction_they_are_facing", "unknown"),
        "number_of_people_in_line":   int(ppl.get("number_of_people") or 0),

        "line_purpose":               gt.get("line_purpose", "unknown"),
        "start_person_description":   ppl.get("start_person_description", "unknown"),
        "end_person_description":     ppl.get("end_person_description", "unknown"),
        "counter_person_description": ppl.get("counter_person_description", "unknown"),

        "boundary_present":           bool_to_yesno(bnd.get("boundary_present", "no")),
        "boundary_types":             bnd.get("boundary_types", "none"),

        "end_of_line_visible":                    eol.get("visible", "no"),
        "end_of_line_location_if_visible":        eol.get("location_if_visible", "N/A"),
        "direction_to_turn_to_see_end_if_not_visible": eol.get("direction_to_turn_if_not_visible", "N/A"),

        "start_of_line_visible":                  sol.get("visible", "no"),
        "start_of_line_location_if_visible":      sol.get("location_if_visible", "N/A"),
        "direction_to_turn_to_see_start_if_not_visible": sol.get("direction_to_turn_if_not_visible", "N/A"),

        "line_completeness":          gt.get("line_completeness", "partial"),
    }

    # Ensure every expected key is present
    for k in FLAT_KEYS:
        flat.setdefault(k, "N/A")

    return flat

# ---------- New helper: write image paths for GT JSON files ----------
def _find_image_path_in_gt_obj(gt_obj):
    """
    Try to find an image path stored inside a GT JSON object.
    Searches common key names, recursively.
    """
    candidate_keys = {
        "image_path", "img_path", "path", "file_path", "filename", "file_name",
        "image", "img", "source_image", "source_path"
    }
    valid_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")

    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in candidate_keys and isinstance(v, str) and v.lower().endswith(valid_exts):
                    return os.path.abspath(v)
            for v in obj.values():
                out = _search(v)
                if out:
                    return out
        elif isinstance(obj, list):
            for v in obj:
                out = _search(v)
                if out:
                    return out
        return None

    return _search(gt_obj)

def write_image_paths_with_gt_json(gt_dict, output_txt, image_root=None):
    """
    Write one image path per line for every GT JSON entry.

    Priority:
      1) use image path stored inside the GT JSON, if present
      2) if image_root is provided, find a file with the same basename under image_root
      3) otherwise write the basename only and warn
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
    unresolved = []

    with open(output_txt, "w", encoding="utf-8") as f:
        for key in sorted(gt_dict.keys()):
            gt_obj = gt_dict[key]
            image_path = _find_image_path_in_gt_obj(gt_obj)

            if image_path is None and image_root is not None:
                found = None
                for dirpath, _, filenames in os.walk(image_root):
                    for fname in filenames:
                        stem, ext = os.path.splitext(fname)
                        if stem == key and ext.lower() in valid_exts:
                            found = os.path.abspath(os.path.join(dirpath, fname))
                            break
                    if found is not None:
                        break
                image_path = found

            if image_path is None:
                unresolved.append(key)
                f.write(f"{key}\n")
            else:
                f.write(f"{image_path}\n")

    print(f"Wrote image paths for {len(gt_dict)} GT JSON files to: {os.path.abspath(output_txt)}")
    if unresolved:
        print(f"[warn] Could not resolve full image path for {len(unresolved)} entries; wrote basename only.")
        for k in unresolved[:20]:
            print(f"  unresolved: {k}")
        if len(unresolved) > 20:
            print("  ...")

# ---------- Evaluation ----------
def evaluate_accuracy_with_mismatches(results: dict, gt_dict: dict, gt_paths: dict):
    """
    results: { image_name(no ext or with ext) : GPT flat dict }
    gt_dict: { image_name(no ext)            : GT nested dict }
    gt_paths:{ image_name(no ext)            : path to GT json }

    Returns:
      - per-key accuracy dict
      - overall accuracy
      - mismatches: dict[field] = list of (image_name, gpt_value, gt_value, gt_path)
    """
    # Normalize result keys to match GT keys (strip extension if needed)
    def norm_key(k):
        return os.path.splitext(k)[0]

    correct_counts = {k: 0 for k in FLAT_KEYS}
    total_counts   = {k: 0 for k in FLAT_KEYS}
    mismatches     = {k: [] for k in FLAT_KEYS}

    common = set(map(norm_key, results.keys())) & set(gt_dict.keys())

    for img_norm in sorted(common):
        # get original result key (might include extension)
        # pick the first key in results whose basename (no ext) matches
        rkey = next(k for k in results.keys() if norm_key(k) == img_norm)
        gpt_res = results[rkey]
        gt_res  = normalize_gt_to_flat(gt_dict[img_norm])

        for k in FLAT_KEYS:
            total_counts[k] += 1
            gv = str(gpt_res.get(k)).strip().lower()
            tv = str(gt_res.get(k)).strip().lower()
            if gv == tv:
                correct_counts[k] += 1
            else:
                mismatches[k].append((rkey, gpt_res.get(k), gt_res.get(k), gt_paths.get(img_norm, "N/A")))

    accuracies = {k: (correct_counts[k] / total_counts[k] if total_counts[k] else None) for k in FLAT_KEYS}
    overall = sum(correct_counts.values()) / sum(total_counts.values()) if sum(total_counts.values()) else None
    return accuracies, overall, mismatches

# ---------- Pretty print the three requested mismatch lists ----------
def print_specific_mismatches(mismatches):
    fields = [
        "line_facing_direction",
        "end_of_line_visible",
        "start_of_line_visible",
    ]
    for f in fields:
        print(f"\nIncorrect {f}:")
        if not mismatches.get(f):
            print("  (none)")
            continue
        for img_name, gpt_v, gt_v, gt_path in mismatches[f]:
            print(f"  {img_name} | GPT={gpt_v!r} | GT={gt_v!r} | GT JSON: {gt_path}")

if __name__ == "__main__":
    # Example usage
    results = read_result_file("gpt_line_test.jsonl")
    print(f"Loaded {len(results)} records")

    folder = "/scratch/ds5725/linefinder/LineFinder/GT_json"
    gt_dict, gt_paths = load_all_json_recursive_with_paths(folder)
    print(f"Loaded {len(gt_dict)} JSON files")

    # NEW: write all image paths with GT JSON to a txt file.
    # If your GT JSON already stores image paths, image_root is not needed.
    # Otherwise set image_root to your image folder and the script will try to
    # match by basename, e.g. IMG_0001.json -> IMG_0001.jpg/png/...
    write_image_paths_with_gt_json(
        gt_dict,
        output_txt="images_with_gt_json.txt",
        image_root=None,  # e.g. "/scratch/ds5725/linefinder/LineFinder/Images"
    )
    exit()
    # Optional: quick diff to see mismatches
    def diff_records(a: dict, b: dict):
        diffs = {}
        for k in FLAT_KEYS:
            if a.get(k) != b.get(k):
                diffs[k] = (a.get(k), b.get(k))
        return diffs

    acc, overall, mismatches = evaluate_accuracy_with_mismatches(results, gt_dict, gt_paths)

    print("Per-key accuracy:")
    for k, v in acc.items():
        print(f"{k:40s}: {v:.2%}")
    print(f"Overall accuracy: {overall:.2%}")
    print_specific_mismatches(mismatches)
