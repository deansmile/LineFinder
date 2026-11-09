#!/usr/bin/env python3
import os
import json
import math
import csv
import argparse
from typing import Dict, List, Optional

QUESTION = "Number of people in line"
TOO_HARD_KEY = "Is this image too hard?()".replace("()", "")  # avoids typo edits


def safe_get(d: Dict, key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def parse_int_value(value) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(round(value))

    s = str(value).strip()
    if s == "":
        return None

    try:
        if "." in s:
            return int(round(float(s)))
        return int(s)
    except ValueError:
        return None


def load_json(json_path: str) -> Optional[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {json_path}: {e}")
        return None


def collect_json_files(folder: str) -> List[str]:
    json_files = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".json"):
                json_files.append(os.path.join(root, name))
    return sorted(json_files)


def build_gt_index(json_files: List[str]) -> Dict[str, str]:
    """
    Build mapping:
        image_id -> json_path

    image_id is taken from:
      1. data["image_id"] if present
      2. basename(json_path) without .json
    """
    index = {}

    for json_path in json_files:
        data = load_json(json_path)
        if data is None:
            continue

        image_id = safe_get(data, "image_id", None)
        if not image_id:
            image_id = os.path.splitext(os.path.basename(json_path))[0]

        image_id = str(image_id).strip()
        if not image_id:
            continue

        if image_id in index:
            print(f"[WARN] Duplicate image_id '{image_id}' found. Keeping first: {index[image_id]}")
            continue

        index[image_id] = json_path

    return index


def load_predictions_csv(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_results(
    results: List[Dict],
    total_predictions: int,
    skipped_too_hard: int,
    skipped_missing_gt: int,
    skipped_missing_prediction: int,
    skipped_missing_field: int,
    skipped_invalid_number: int,
) -> Dict:
    n = len(results)

    if n == 0:
        return {
            "total_predictions": total_predictions,
            "evaluated": 0,
            "skipped_too_hard": skipped_too_hard,
            "skipped_missing_gt": skipped_missing_gt,
            "skipped_missing_prediction": skipped_missing_prediction,
            "skipped_missing_field": skipped_missing_field,
            "skipped_invalid_number": skipped_invalid_number,
            "metrics": {},
            "per_image": results,
        }

    abs_errors = [r["abs_error"] for r in results]
    signed_errors = [r["signed_error"] for r in results]
    sq_errors = [r["signed_error"] ** 2 for r in results]
    rel_errors = [r["relative_error"] for r in results if r["relative_error"] is not None]

    metrics = {
        "num_samples": n,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(sq_errors) / n),
        "mean_signed_error": sum(signed_errors) / n,
        "mean_relative_error": (sum(rel_errors) / len(rel_errors)) if rel_errors else None,
        "exact_match_accuracy": sum(r["is_exact"] for r in results) / n,
        "accuracy_within_1": sum(r["within_1"] for r in results) / n,
        "accuracy_within_2": sum(r["within_2"] for r in results) / n,
        "undercount_rate": sum(r["is_undercount"] for r in results) / n,
        "overcount_rate": sum(r["is_overcount"] for r in results) / n,
        "num_undercount": sum(r["is_undercount"] for r in results),
        "num_overcount": sum(r["is_overcount"] for r in results),
        "num_exact": sum(r["is_exact"] for r in results),
    }

    return {
        "total_predictions": total_predictions,
        "evaluated": n,
        "skipped_too_hard": skipped_too_hard,
        "skipped_missing_gt": skipped_missing_gt,
        "skipped_missing_prediction": skipped_missing_prediction,
        "skipped_missing_field": skipped_missing_field,
        "skipped_invalid_number": skipped_invalid_number,
        "metrics": metrics,
        "per_image": results,
    }


def evaluate_csv_against_gt(
    csv_rows: List[Dict],
    gt_index: Dict[str, str],
    ignore_too_hard: bool = True,
    pred_col: str = "OL_number_of_people",
    status_col: str = "status",
    require_ok_status: bool = True,
) -> Dict:
    results = []

    skipped_too_hard = 0
    skipped_missing_gt = 0
    skipped_missing_prediction = 0
    skipped_missing_field = 0
    skipped_invalid_number = 0

    for row in csv_rows:
        image_id = str(row.get("image_id", "")).strip()

        if not image_id:
            image_path = str(row.get("image_path", "")).strip()
            if image_path:
                image_id = os.path.splitext(os.path.basename(image_path))[0]

        if not image_id:
            skipped_missing_prediction += 1
            print(f"[WARN] Missing image_id in CSV row: {row}")
            continue

        if require_ok_status:
            status = str(row.get(status_col, "")).strip().lower()
            if status != "ok":
                skipped_missing_prediction += 1
                print(f"[WARN] Skipping {image_id} due to status={repr(row.get(status_col, ''))}")
                continue

        pred_raw = row.get(pred_col, None)
        if pred_raw is None or str(pred_raw).strip() == "":
            skipped_missing_prediction += 1
            print(f"[WARN] Missing prediction for {image_id}")
            continue

        json_path = gt_index.get(image_id)
        if json_path is None:
            skipped_missing_gt += 1
            print(f"[WARN] No GT JSON found for image_id={image_id}")
            continue

        data = load_json(json_path)
        if data is None:
            skipped_missing_gt += 1
            continue

        annotation = safe_get(data, "annotation", {})

        too_hard = str(safe_get(annotation, TOO_HARD_KEY, "no")).strip().lower()
        if ignore_too_hard and too_hard == "yes":
            skipped_too_hard += 1
            continue

        gt_raw = safe_get(annotation, QUESTION, None)
        if gt_raw is None:
            skipped_missing_field += 1
            print(f"[WARN] Missing '{QUESTION}' in {json_path}")
            continue

        gt = parse_int_value(gt_raw)
        pred = parse_int_value(pred_raw)

        if gt is None or pred is None:
            skipped_invalid_number += 1
            print(
                f"[WARN] Invalid number for {image_id} | "
                f"gt={repr(gt_raw)} pred={repr(pred_raw)}"
            )
            continue

        signed_error = pred - gt
        abs_error = abs(signed_error)
        relative_error = abs_error / gt if gt != 0 else (0.0 if pred == 0 else None)

        results.append(
            {
                "json_path": json_path,
                "image_id": image_id,
                "image_name": safe_get(data, "image_name", ""),
                "gt": gt,
                "pred": pred,
                "signed_error": signed_error,
                "abs_error": abs_error,
                "relative_error": relative_error,
                "is_exact": int(pred == gt),
                "within_1": int(abs_error <= 1),
                "within_2": int(abs_error <= 2),
                "is_undercount": int(pred < gt),
                "is_overcount": int(pred > gt),
            }
        )

    return summarize_results(
        results=results,
        total_predictions=len(csv_rows),
        skipped_too_hard=skipped_too_hard,
        skipped_missing_gt=skipped_missing_gt,
        skipped_missing_prediction=skipped_missing_prediction,
        skipped_missing_field=skipped_missing_field,
        skipped_invalid_number=skipped_invalid_number,
    )


def print_summary(name: str, summary: Dict):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total prediction rows:       {summary['total_predictions']}")
    print(f"Evaluated samples:           {summary['evaluated']}")
    print(f"Skipped too hard:            {summary['skipped_too_hard']}")
    print(f"Skipped missing GT json:     {summary['skipped_missing_gt']}")
    print(f"Skipped missing prediction:  {summary['skipped_missing_prediction']}")
    print(f"Skipped missing field:       {summary['skipped_missing_field']}")
    print(f"Skipped invalid number:      {summary['skipped_invalid_number']}")

    metrics = summary["metrics"]
    if not metrics:
        print("No valid samples to evaluate.\n")
        return

    print("\nDataset-level metrics for 'Number of people in line':")
    print(f"MAE:                         {metrics['mae']:.4f}")
    print(f"RMSE:                        {metrics['rmse']:.4f}")
    print(f"Mean signed error:           {metrics['mean_signed_error']:.4f}")
    if metrics["mean_relative_error"] is None:
        print("Mean relative error:         N/A")
    else:
        print(f"Mean relative error:         {metrics['mean_relative_error']:.4f}")

    print(f"Exact match accuracy:        {metrics['exact_match_accuracy']:.4f}")
    print(f"Accuracy within ±1:          {metrics['accuracy_within_1']:.4f}")
    print(f"Accuracy within ±2:          {metrics['accuracy_within_2']:.4f}")
    print(f"Undercount rate:             {metrics['undercount_rate']:.4f}")
    print(f"Overcount rate:              {metrics['overcount_rate']:.4f}")
    print(f"Number of undercounts:       {metrics['num_undercount']}")
    print(f"Number of overcounts:        {metrics['num_overcount']}")
    print(f"Number of exact matches:     {metrics['num_exact']}")
    print()


def save_detailed_csv(summary: Dict, csv_path: str):
    fieldnames = [
        "json_path",
        "image_id",
        "image_name",
        "gt",
        "pred",
        "signed_error",
        "abs_error",
        "relative_error",
        "is_exact",
        "within_1",
        "within_2",
        "is_undercount",
        "is_overcount",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["per_image"]:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OL_number_of_people.csv against GT annotations."
    )
    parser.add_argument(
        "--pred-csv",
        type=str,
        default="OL_number_of_people.csv",
        help="Path to OL_number_of_people.csv",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            "/mnt/NAS/home/ds5725/linefinder/GT_json/GT_json_luna",
            "/mnt/NAS/home/ds5725/linefinder/GT_json/GT_json_olivia",
        ],
        help="Folders containing GT JSON files.",
    )
    parser.add_argument(
        "--include-too-hard",
        action="store_true",
        help="Include samples annotated as too hard.",
    )
    parser.add_argument(
        "--include-non-ok-status",
        action="store_true",
        help="Include CSV rows whose status is not 'ok'.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save per-image detailed results CSV.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pred_csv):
        print(f"[ERROR] Prediction CSV not found: {args.pred_csv}")
        return

    ignore_too_hard = not args.include_too_hard
    require_ok_status = not args.include_non_ok_status

    all_json_files = []
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"[WARN] Folder does not exist: {folder}")
            continue
        all_json_files.extend(collect_json_files(folder))

    if not all_json_files:
        print("No valid GT folders found.")
        return

    gt_index = build_gt_index(all_json_files)
    print(f"Indexed {len(gt_index)} GT JSON files.")

    csv_rows = load_predictions_csv(args.pred_csv)
    print(f"Loaded {len(csv_rows)} prediction rows from {args.pred_csv}")

    summary = evaluate_csv_against_gt(
        csv_rows=csv_rows,
        gt_index=gt_index,
        ignore_too_hard=ignore_too_hard,
        pred_col="OL_number_of_people",
        status_col="status",
        require_ok_status=require_ok_status,
    )

    print_summary("Results for OL_number_of_people.csv", summary)

    if args.save_csv is not None:
        save_detailed_csv(summary, args.save_csv)
        print(f"Saved detailed CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()