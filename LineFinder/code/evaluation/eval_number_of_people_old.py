#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
import json
import math
import csv
import argparse
from typing import Dict, List, Optional


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


def find_gt_jsons(gt_root: str) -> Dict[str, str]:
    """
    Recursively build a mapping:
        image_id (basename without .json) -> full json path
    """
    mapping = {}
    for root, _, files in os.walk(gt_root):
        for name in files:
            if name.lower().endswith(".json"):
                image_id = os.path.splitext(name)[0]
                full_path = os.path.join(root, name)
                if image_id in mapping:
                    print(f"[WARN] Duplicate GT json for {image_id}:")
                    print(f"       Keeping:  {mapping[image_id]}")
                    print(f"       Ignoring: {full_path}")
                else:
                    mapping[image_id] = full_path
    return mapping


def summarize_results(
    results: List[Dict],
    total_rows: int,
    skipped_status: int,
    skipped_missing_gt: int,
    skipped_missing_field: int,
    skipped_invalid_number: int,
) -> Dict:
    n = len(results)

    if n == 0:
        return {
            "total_rows": total_rows,
            "evaluated": 0,
            "skipped_status": skipped_status,
            "skipped_missing_gt": skipped_missing_gt,
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
        "total_rows": total_rows,
        "evaluated": n,
        "skipped_status": skipped_status,
        "skipped_missing_gt": skipped_missing_gt,
        "skipped_missing_field": skipped_missing_field,
        "skipped_invalid_number": skipped_invalid_number,
        "metrics": metrics,
        "per_image": results,
    }


def print_summary(summary: Dict):
    print("=" * 80)
    print("Results on OLD data")
    print("=" * 80)
    print(f"Total csv rows:              {summary['total_rows']}")
    print(f"Evaluated samples:           {summary['evaluated']}")
    print(f"Skipped bad status:          {summary['skipped_status']}")
    print(f"Skipped missing GT json:     {summary['skipped_missing_gt']}")
    print(f"Skipped missing field:       {summary['skipped_missing_field']}")
    print(f"Skipped invalid number:      {summary['skipped_invalid_number']}")

    metrics = summary["metrics"]
    if not metrics:
        print("No valid samples to evaluate.")
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


def save_detailed_csv(summary: Dict, csv_path: str):
    fieldnames = [
        "image_id",
        "image_path",
        "gt_json_path",
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


def evaluate_old_data(result_csv: str, gt_root: str) -> Dict:
    gt_map = find_gt_jsons(gt_root)

    results = []
    total_rows = 0
    skipped_status = 0
    skipped_missing_gt = 0
    skipped_missing_field = 0
    skipped_invalid_number = 0

    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"image_id", "image_path", "OL_number_of_people", "status"}
        missing_cols = required_cols - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

        for row in reader:
            total_rows += 1

            image_id = str(row["image_id"]).strip()
            image_path = str(row["image_path"]).strip()
            pred_raw = row["OL_number_of_people"]
            status = str(row["status"]).strip().lower()

            if status != "ok":
                skipped_status += 1
                continue

            gt_json_path = gt_map.get(image_id)
            if gt_json_path is None:
                skipped_missing_gt += 1
                print(f"[WARN] Missing GT json for image_id={image_id}")
                continue

            data = load_json(gt_json_path)
            if data is None:
                skipped_missing_gt += 1
                continue

            people = data.get("people", {})
            gt_raw = people.get("number_of_people", None)

            if gt_raw is None or pred_raw is None:
                skipped_missing_field += 1
                print(f"[WARN] Missing people.number_of_people for image_id={image_id}")
                continue

            gt = parse_int_value(gt_raw)
            pred = parse_int_value(pred_raw)

            if gt is None or pred is None:
                skipped_invalid_number += 1
                print(
                    f"[WARN] Invalid number for image_id={image_id} | "
                    f"gt={repr(gt_raw)} pred={repr(pred_raw)}"
                )
                continue

            signed_error = pred - gt
            abs_error = abs(signed_error)
            relative_error = abs_error / gt if gt != 0 else (0.0 if pred == 0 else None)

            results.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "gt_json_path": gt_json_path,
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
        total_rows=total_rows,
        skipped_status=skipped_status,
        skipped_missing_gt=skipped_missing_gt,
        skipped_missing_field=skipped_missing_field,
        skipped_invalid_number=skipped_invalid_number,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate old-data number-of-people predictions against GT jsons."
    )
    parser.add_argument(
        "--result-csv",
        type=str,
        default=lf_path("code/OLD_number_of_people.csv"),
        help="Path to OLD_number_of_people.csv",
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        default=lf_path("GT_json"),
        help="Root directory containing GT json files recursively.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save per-image evaluation results.",
    )
    args = parser.parse_args()

    summary = evaluate_old_data(
        result_csv=args.result_csv,
        gt_root=args.gt_root,
    )

    print_summary(summary)

    if args.save_csv:
        save_detailed_csv(summary, args.save_csv)
        print(f"\nSaved detailed CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()