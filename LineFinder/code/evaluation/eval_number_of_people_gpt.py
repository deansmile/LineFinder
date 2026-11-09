#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
import json
import math
import argparse
from typing import Dict, List, Optional

QUESTION = "Number of people in line"
TOO_HARD_KEY = "Is this image too hard?()".replace("()", "")


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


def load_gpt_predictions(jsonl_path: str) -> Dict[str, Dict]:
    """
    Expected format:
    line 1: supermarket_001.jpg
    line 2: {"Number of people in line": 18, ...}
    line 3: supermarket_002.jpg
    line 4: {...}

    Returns:
        {
            "supermarket_001": {...},
            "supermarket_002": {...},
        }
    """
    predictions = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        image_line = lines[i]

        if i + 1 >= len(lines):
            print(f"[WARN] Missing JSON prediction after image line: {image_line}")
            break

        pred_line = lines[i + 1]

        image_name = os.path.basename(image_line)
        image_id = os.path.splitext(image_name)[0]

        try:
            pred = json.loads(pred_line)
        except Exception as e:
            print(f"[WARN] Failed to parse prediction JSON for {image_name}: {e}")
            i += 2
            continue

        predictions[image_id] = pred
        i += 2

    return predictions


def evaluate_files(
    json_files: List[str],
    gpt_predictions: Dict[str, Dict],
    ignore_too_hard: bool = True,
) -> Dict:
    results = []

    skipped_too_hard = 0
    skipped_missing_field = 0
    skipped_invalid_number = 0
    skipped_missing_prediction = 0

    for json_path in json_files:
        data = load_json(json_path)
        if data is None:
            continue

        annotation = safe_get(data, "annotation", {})

        image_id = safe_get(data, "image_id", os.path.splitext(os.path.basename(json_path))[0])
        image_name = safe_get(data, "image_name", "")

        prediction = gpt_predictions.get(image_id)
        if prediction is None and image_name:
            prediction = gpt_predictions.get(os.path.splitext(os.path.basename(image_name))[0])

        if prediction is None:
            skipped_missing_prediction += 1
            print(f"[WARN] Missing GPT prediction for {image_id}")
            continue

        too_hard = str(safe_get(annotation, TOO_HARD_KEY, "no")).strip().lower()
        if ignore_too_hard and too_hard == "yes":
            skipped_too_hard += 1
            continue

        gt_raw = safe_get(annotation, QUESTION, None)
        pred_raw = safe_get(prediction, QUESTION, None)

        if gt_raw is None or pred_raw is None:
            skipped_missing_field += 1
            print(f"[WARN] Missing '{QUESTION}' in {json_path}")
            continue

        gt = parse_int_value(gt_raw)
        pred = parse_int_value(pred_raw)

        if gt is None or pred is None:
            skipped_invalid_number += 1
            print(
                f"[WARN] Invalid number in {json_path} | "
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
                "image_name": image_name,
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
        total_jsons=len(json_files),
        skipped_too_hard=skipped_too_hard,
        skipped_missing_field=skipped_missing_field,
        skipped_invalid_number=skipped_invalid_number,
        skipped_missing_prediction=skipped_missing_prediction,
    )


def summarize_results(
    results: List[Dict],
    total_jsons: int,
    skipped_too_hard: int,
    skipped_missing_field: int,
    skipped_invalid_number: int,
    skipped_missing_prediction: int,
) -> Dict:
    n = len(results)

    if n == 0:
        return {
            "total_jsons": total_jsons,
            "evaluated": 0,
            "skipped_too_hard": skipped_too_hard,
            "skipped_missing_field": skipped_missing_field,
            "skipped_invalid_number": skipped_invalid_number,
            "skipped_missing_prediction": skipped_missing_prediction,
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
        "total_jsons": total_jsons,
        "evaluated": n,
        "skipped_too_hard": skipped_too_hard,
        "skipped_missing_field": skipped_missing_field,
        "skipped_invalid_number": skipped_invalid_number,
        "skipped_missing_prediction": skipped_missing_prediction,
        "metrics": metrics,
        "per_image": results,
    }


def print_summary(name: str, summary: Dict):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total json files found:      {summary['total_jsons']}")
    print(f"Evaluated samples:           {summary['evaluated']}")
    print(f"Skipped too hard:            {summary['skipped_too_hard']}")
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
    import csv

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


def merge_summaries(summaries: List[Dict]) -> Dict:
    merged_results = []
    total_jsons = 0
    skipped_too_hard = 0
    skipped_missing_field = 0
    skipped_invalid_number = 0
    skipped_missing_prediction = 0

    for s in summaries:
        merged_results.extend(s["per_image"])
        total_jsons += s["total_jsons"]
        skipped_too_hard += s["skipped_too_hard"]
        skipped_missing_field += s["skipped_missing_field"]
        skipped_invalid_number += s["skipped_invalid_number"]
        skipped_missing_prediction += s["skipped_missing_prediction"]

    return summarize_results(
        results=merged_results,
        total_jsons=total_jsons,
        skipped_too_hard=skipped_too_hard,
        skipped_missing_field=skipped_missing_field,
        skipped_invalid_number=skipped_invalid_number,
        skipped_missing_prediction=skipped_missing_prediction,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPT 'Number of people in line' predictions against annotations."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            lf_path("GT_json/GT_json_luna"),
            lf_path("GT_json/GT_json_olivia"),
        ],
        help="Folders containing GT JSON files.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="gpt_line_luna_olivia.jsonl",
        help="Path to GPT predictions JSONL file.",
    )
    parser.add_argument(
        "--include-too-hard",
        action="store_true",
        help="Include samples annotated as 'Is this image too hard?' == yes",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save per-image detailed results as CSV for the combined set.",
    )
    args = parser.parse_args()

    ignore_too_hard = not args.include_too_hard
    gpt_predictions = load_gpt_predictions(args.predictions_jsonl)

    all_folder_summaries = []

    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"[WARN] Folder does not exist: {folder}")
            continue

        json_files = collect_json_files(folder)
        summary = evaluate_files(
            json_files,
            gpt_predictions=gpt_predictions,
            ignore_too_hard=ignore_too_hard,
        )
        all_folder_summaries.append(summary)

    if not all_folder_summaries:
        print("No valid folders found.")
        return

    combined = merge_summaries(all_folder_summaries)
    print_summary("Combined results", combined)

    if args.save_csv is not None:
        save_detailed_csv(combined, args.save_csv)
        print(f"Saved detailed CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()