import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
import json
from typing import Dict, Tuple, List

EVAL_FIELDS = [
    "What direction is the line facing",
    "Do we see the end of line?",
    "If yes, where is the end of the line in the image?",
    "If not, in what direction should you move camera to see the end of the line?",
    "Do we see the start of line?",
    "If yes, where is the start of the line in the image?",
    "If not, in what direction should you move camera to see the start of the line?",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")

def is_too_hard(gt_obj):
    ann = gt_obj.get("annotation", {})
    val = str(ann.get("Is this image too hard?", "no")).strip().lower()
    return val == "yes"
    
def normalize_text(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"n/a", "na", "none", "null", ""}:
        return "n/a"
    return s.lower()


def strip_image_extension(filename: str) -> str:
    base = os.path.basename(filename.strip())
    for ext in IMAGE_EXTS:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def read_result_file(path: str) -> Dict[str, dict]:
    """
    Reads a result file like:

    supermarket_001.jpg
    { ...json... }

    or fenced json blocks:

    supermarket_001.jpg
    ```json
    { ...json... }
    ```

    Returns: {image_id_without_ext: parsed_json}
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
                results[image_id] = json.loads("".join(json_lines))
            except Exception as e:
                print(f"[warn] failed to parse record for {image_id}: {e}")
        else:
            i += 1

    return results


def load_gt_json_dirs(folders: List[str]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Load GT json files from multiple folders.
    Returns:
      gt_dict  : {image_id: parsed_gt_json}
      gt_paths : {image_id: absolute_json_path}
    """
    gt_dict = {}
    gt_paths = {}

    for folder in folders:
        if not os.path.isdir(folder):
            print(f"[warn] GT folder does not exist: {folder}")
            continue

        for dirpath, _, filenames in os.walk(folder):
            for fname in filenames:
                if not fname.lower().endswith(".json"):
                    continue
                fpath = os.path.join(dirpath, fname)
                key = os.path.splitext(fname)[0]
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        gt_obj = json.load(f)
                    gt_dict[key] = gt_obj
                    gt_paths[key] = os.path.abspath(fpath)
                except Exception as e:
                    print(f"[warn] failed to load {fpath}: {e}")

    return gt_dict, gt_paths


def extract_gt_annotation(gt_obj: dict) -> dict:
    """
    The new GT format stores the target answers in gt_obj['annotation'].
    Fallback to gt_obj['prediction'] only if annotation is missing.
    """
    if isinstance(gt_obj.get("annotation"), dict):
        return gt_obj["annotation"]
    if isinstance(gt_obj.get("prediction"), dict):
        return gt_obj["prediction"]
    return gt_obj


def evaluate_accuracy_with_mismatches(results: Dict[str, dict], gt_dict: Dict[str, dict], gt_paths: Dict[str, str]):
    correct_counts = {k: 0 for k in EVAL_FIELDS}
    total_counts = {k: 0 for k in EVAL_FIELDS}
    mismatches = {k: [] for k in EVAL_FIELDS}

    common_ids = sorted(set(results.keys()) & set(gt_dict.keys()))

    hard_count = 0
    not_hard_count = 0

    for image_id in common_ids:
        gt_obj = gt_dict[image_id]

        # 🔴 NEW: filter "too hard"
        if is_too_hard(gt_obj):
            hard_count += 1
            continue
        else:
            not_hard_count += 1

        pred = results[image_id]
        gt_ann = extract_gt_annotation(gt_obj)

        for field in EVAL_FIELDS:
            pred_value = pred.get(field, "N/A")
            gt_value = gt_ann.get(field, "N/A")

            total_counts[field] += 1
            if normalize_text(pred_value) == normalize_text(gt_value):
                correct_counts[field] += 1
            else:
                mismatches[field].append(
                    {
                        "image_id": image_id,
                        "prediction": pred_value,
                        "ground_truth": gt_value,
                        "gt_json": gt_paths.get(image_id, "N/A"),
                    }
                )

    accuracies = {
        field: (correct_counts[field] / total_counts[field] if total_counts[field] else None)
        for field in EVAL_FIELDS
    }

    overall_correct = sum(correct_counts.values())
    overall_total = sum(total_counts.values())
    overall_accuracy = overall_correct / overall_total if overall_total else None

    return accuracies, overall_accuracy, mismatches, common_ids, hard_count, not_hard_count


def print_summary(accuracies, overall_accuracy, common_ids, results, gt_dict):
    print(f"Loaded {len(results)} result records")
    print(f"Loaded {len(gt_dict)} GT json files")
    print(f"Matched {len(common_ids)} images between results and GT")
    print()
    print("Per-field accuracy:")
    for field in EVAL_FIELDS:
        value = accuracies[field]
        if value is None:
            print(f"  {field}: N/A")
        else:
            print(f"  {field}: {value:.2%}")
    print()
    if overall_accuracy is None:
        print("Overall accuracy: N/A")
    else:
        print(f"Overall accuracy: {overall_accuracy:.2%}")


def print_mismatches(mismatches):
    print("\nDetailed mismatches:")
    for field in EVAL_FIELDS:
        print(f"\n{field}")
        print("-" * len(field))
        if not mismatches[field]:
            print("  (none)")
            continue
        for item in mismatches[field]:
            print(
                f"  {item['image_id']} | pred={item['prediction']!r} | "
                f"gt={item['ground_truth']!r} | GT JSON: {item['gt_json']}"
            )


def main():
    result_path = "gpt_line_luna_olivia.jsonl"
    gt_folders = [
        lf_path("GT_json/GT_json_luna"),
        lf_path("GT_json/GT_json_olivia"),
    ]

    results = read_result_file(result_path)
    gt_dict, gt_paths = load_gt_json_dirs(gt_folders)

    accuracies, overall_accuracy, mismatches, common_ids, hard_count, not_hard_count = evaluate_accuracy_with_mismatches(
        results, gt_dict, gt_paths
    )

    print_summary(accuracies, overall_accuracy, common_ids, results, gt_dict)
    print(f"\nHard images: {hard_count}")
    print(f"Not hard images (used for evaluation): {not_hard_count}\n") 
    # print_mismatches(mismatches)


if __name__ == "__main__":
    main()
