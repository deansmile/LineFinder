#!/usr/bin/env python3
import os
import json
import math
import csv
import argparse
from typing import Dict, List, Optional


GPT_COUNT_KEY = "number_of_people_in_line"


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


def clean_json_block(text: str) -> str:
    """
    Removes markdown fences like:
    ```json
    {...}
    ```
    """
    s = text.strip()

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    return s


def load_gpt_predictions(jsonl_path: str) -> Dict[str, Dict]:
    """
    Expected format:
    line 1: IMG_5954.JPG
    line 2..k: ```json ... ```
    line k+1: IMG_5955.JPG
    ...

    Returns:
        {
            "IMG_5954": {...},
            "IMG_5955": {...},
        }
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    predictions = {}
    i = 0
    n = len(raw_lines)

    while i < n:
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue

        # image filename line
        image_name = os.path.basename(line)
        image_id = os.path.splitext(image_name)[0]
        i += 1

        # skip empty lines
        while i < n and not raw_lines[i].strip():
            i += 1

        if i >= n:
            print(f"[WARN] Missing prediction block after image line: {image_name}")
            break

        # collect prediction block
        block_lines = []

        if raw_lines[i].strip().startswith("```"):
            # fenced JSON block
            block_lines.append(raw_lines[i])
            i += 1
            while i < n:
                block_lines.append(raw_lines[i])
                if raw_lines[i].strip() == "```":
                    i += 1
                    break
                i += 1
        else:
            # single-line JSON fallback
            block_lines.append(raw_lines[i])
            i += 1

        pred_text = "\n".join(block_lines)
        pred_text = clean_json_block(pred_text)

        try:
            pred = json.loads(pred_text)
        except Exception as e:
            print(f"[WARN] Failed to parse prediction JSON for {image_name}: {e}")
            continue

        predictions[image_id] = pred

    return predictions


def summarize_results(
    results: List[Dict],
    total_items: int,
    skipped_missing_gt: int,
    skipped_missing_prediction: int,
    skipped_missing_field: int,
    skipped_invalid_number: int,
) -> Dict:
    n = len(results)

    if n == 0:
        return {
            "total_items": total_items,
            "evaluated": 0,
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
        "total_items": total_items,
        "evaluated": n,
        "skipped_missing_gt": skipped_missing_gt,
        "skipped_missing_prediction": skipped_missing_prediction,
        "skipped_missing_field": skipped_missing_field,
        "skipped_invalid_number": skipped_invalid_number,
        "metrics": metrics,
        "per_image": results,
    }


def print_summary(summary: Dict):
    print("=" * 80)
    print("Results on OLD data (GPT)")
    print("=" * 80)
    print(f"Total prediction items:      {summary['total_items']}")
    print(f"Evaluated samples:           {summary['evaluated']}")
    print(f"Skipped missing GT json:     {summary['skipped_missing_gt']}")
    print(f"Skipped missing prediction:  {summary['skipped_missing_prediction']}")
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


def evaluate_old_data(predictions_jsonl: str, gt_root: str) -> Dict:
    gt_map = find_gt_jsons(gt_root)
    gpt_predictions = load_gpt_predictions(predictions_jsonl)

    results = []
    skipped_missing_gt = 0
    skipped_missing_prediction = 0
    skipped_missing_field = 0
    skipped_invalid_number = 0

    # evaluate on all prediction items
    for image_id, prediction in gpt_predictions.items():
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
        pred_raw = prediction.get(GPT_COUNT_KEY, None)

        if gt_raw is None or pred_raw is None:
            skipped_missing_field += 1
            print(
                f"[WARN] Missing count field for image_id={image_id} | "
                f"gt_field=people.number_of_people pred_field={GPT_COUNT_KEY}"
            )
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

    # count GTs with no prediction if useful for debugging
    for image_id in gt_map:
        if image_id not in gpt_predictions:
            skipped_missing_prediction += 1

    return summarize_results(
        results=results,
        total_items=len(gpt_predictions),
        skipped_missing_gt=skipped_missing_gt,
        skipped_missing_prediction=skipped_missing_prediction,
        skipped_missing_field=skipped_missing_field,
        skipped_invalid_number=skipped_invalid_number,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate old-data GPT number-of-people predictions against GT jsons."
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="../gpt_line_old.jsonl",
        help="Path to gpt_line_old.jsonl",
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        default="/mnt/NAS/home/ds5725/linefinder/GT_json",
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
        predictions_jsonl=args.predictions_jsonl,
        gt_root=args.gt_root,
    )

    print_summary(summary)

    if args.save_csv:
        save_detailed_csv(summary, args.save_csv)
        print(f"\nSaved detailed CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()