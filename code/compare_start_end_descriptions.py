import os
import json

# Input folders
INPUT_DIRS = [
    "/mnt/NAS/home/ds5725/linefinder/GT_json/GT_json_luna",
    "/mnt/NAS/home/ds5725/linefinder/GT_json/GT_json_olivia",
]

# Output report
OUTPUT_TXT = "/mnt/NAS/home/ds5725/linefinder/different_start_end_descriptions.txt"


def normalize_text(text):
    """Normalize text for comparison."""
    if text is None:
        return ""
    return str(text).strip().lower()


def collect_json_files(folder):
    """Recursively collect all .json files under a folder."""
    json_files = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".json"):
                json_files.append(os.path.join(root, fname))
    return sorted(json_files)


def main():
    mismatches = []
    total_files = 0

    for input_dir in INPUT_DIRS:
        json_files = collect_json_files(input_dir)

        for json_path in json_files:
            total_files += 1

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                mismatches.append({
                    "image_name": os.path.basename(json_path),
                    "json_path": json_path,
                    "error": f"Failed to read JSON: {e}"
                })
                continue

            image_name = data.get("image_name", os.path.basename(json_path))
            prediction = data.get("prediction", {})
            annotation = data.get("annotation", {})

            pred_start = normalize_text(prediction.get("Description of the person at the start of the line", ""))
            gt_start = normalize_text(annotation.get("Description of the person at the start of the line", ""))

            pred_end = normalize_text(prediction.get("Description of the person at the end of the line", ""))
            gt_end = normalize_text(annotation.get("Description of the person at the end of the line", ""))

            start_diff = pred_start != gt_start
            end_diff = pred_end != gt_end

            if start_diff or end_diff:
                mismatches.append({
                    "image_name": image_name,
                    "json_path": json_path,
                    "start_diff": start_diff,
                    "pred_start": prediction.get("Description of the person at the start of the line", ""),
                    "gt_start": annotation.get("Description of the person at the start of the line", ""),
                    "end_diff": end_diff,
                    "pred_end": prediction.get("Description of the person at the end of the line", ""),
                    "gt_end": annotation.get("Description of the person at the end of the line", ""),
                })

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Total JSON files checked: {total_files}\n")
        f.write(f"Files with different start or end person descriptions: {len(mismatches)}\n\n")

        for item in mismatches:
            f.write("=" * 80 + "\n")
            f.write(f"Image name: {item.get('image_name', '')}\n")
            f.write(f"JSON path: {item.get('json_path', '')}\n")

            if "error" in item:
                f.write(f"Error: {item['error']}\n")
                continue

            if item["start_diff"]:
                f.write("\n[Start person description differs]\n")
                f.write(f"Prediction: {item['pred_start']}\n")
                f.write(f"Annotation: {item['gt_start']}\n")

            if item["end_diff"]:
                f.write("\n[End person description differs]\n")
                f.write(f"Prediction: {item['pred_end']}\n")
                f.write(f"Annotation: {item['gt_end']}\n")

            f.write("\n")

    print(f"Done. Report written to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()