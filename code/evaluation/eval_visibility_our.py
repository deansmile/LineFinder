import csv

GT_CSV = "/mnt/NAS/home/ds5725/linefinder/GT_json_canonical.csv"
PRED_CSV = "/mnt/NAS/home/ds5725/linefinder/code/line_visibility_new.csv"

EVAL_FIELDS = [
    ("end visible", "end_of_line_visible"),
    ("end location", "end_of_line_location_if_visible"),
    ("end direction to turn", "direction_to_turn_to_see_end_if_not_visible"),
    ("start visible", "start_of_line_visible"),
    ("start location", "start_of_line_location_if_visible"),
    ("start direction to turn", "direction_to_turn_to_see_start_if_not_visible"),
]


def normalize_text(x):
    if x is None:
        return "n/a"
    s = str(x).strip()
    if s == "":
        return "n/a"
    s = s.lower()
    if s in {"n/a", "na", "none", "null", "unknown"}:
        return "n/a"
    return s


def normalize_yes_no(x):
    s = normalize_text(x)
    if s in {"yes", "true", "1"}:
        return "yes"
    if s in {"no", "false", "0"}:
        return "no"
    return s


def normalize_facing_location_or_direction(x):
    return normalize_text(x)


def normalize_value(field_name, value):
    if field_name in {"end visible", "start visible"}:
        return normalize_yes_no(value)
    return normalize_facing_location_or_direction(value)


def load_gt_csv(path):
    gt = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                continue
            gt[image_id] = row
    return gt


def load_pred_csv(path):
    preds = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            status = str(row.get("status", "")).strip().lower()
            if not image_id:
                continue
            preds[image_id] = row
            preds[image_id]["_status_norm"] = status
    return preds


def is_too_hard(gt_row):
    val = str(gt_row.get("too_hard", "")).strip().lower()
    return val in {"yes", "true", "1"}


def evaluate():
    gt = load_gt_csv(GT_CSV)
    preds = load_pred_csv(PRED_CSV)

    correct_counts = {gt_field: 0 for gt_field, _ in EVAL_FIELDS}
    total_counts = {gt_field: 0 for gt_field, _ in EVAL_FIELDS}
    mismatches = {gt_field: [] for gt_field, _ in EVAL_FIELDS}

    total_pred_rows = len(preds)
    matched_rows = 0
    skipped_hard = 0
    skipped_bad_status = 0
    skipped_missing_gt = 0

    for image_id, pred_row in preds.items():
        gt_row = gt.get(image_id)
        if gt_row is None:
            skipped_missing_gt += 1
            continue

        matched_rows += 1

        if pred_row.get("_status_norm") != "ok":
            skipped_bad_status += 1
            continue

        if is_too_hard(gt_row):
            skipped_hard += 1
            continue

        for gt_field, pred_field in EVAL_FIELDS:
            gt_val = normalize_value(gt_field, gt_row.get(gt_field, "N/A"))
            pred_val = normalize_value(gt_field, pred_row.get(pred_field, "N/A"))

            total_counts[gt_field] += 1
            if pred_val == gt_val:
                correct_counts[gt_field] += 1
            else:
                mismatches[gt_field].append(
                    {
                        "image_id": image_id,
                        "pred": pred_val,
                        "gt": gt_val,
                    }
                )

    overall_correct = sum(correct_counts.values())
    overall_total = sum(total_counts.values())

    print("===== Evaluation Summary =====")
    print(f"Total prediction rows:       {total_pred_rows}")
    print(f"Matched GT rows:             {matched_rows}")
    print(f"Skipped hard GT rows:        {skipped_hard}")
    print(f"Skipped bad status rows:     {skipped_bad_status}")
    print(f"Skipped missing GT rows:     {skipped_missing_gt}")
    print()

    print("Per-field accuracy:")
    for gt_field, _ in EVAL_FIELDS:
        total = total_counts[gt_field]
        if total == 0:
            print(f"  {gt_field}: N/A")
        else:
            acc = correct_counts[gt_field] / total
            print(f"  {gt_field}: {acc:.2%} ({correct_counts[gt_field]}/{total})")

    print()
    if overall_total == 0:
        print("Overall accuracy: N/A")
    else:
        print(f"Overall accuracy: {overall_correct / overall_total:.2%} ({overall_correct}/{overall_total})")

    return mismatches


if __name__ == "__main__":
    mismatches = evaluate()

    # Uncomment to inspect mismatches
    # for field, items in mismatches.items():
    #     print(f"\n=== {field} mismatches ({len(items)}) ===")
    #     for item in items[:20]:
    #         print(item)