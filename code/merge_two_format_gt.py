import os
import json
import csv
from typing import Dict, Any, List

GT_ROOT = "/mnt/NAS/home/ds5725/linefinder/GT_json"
OUTPUT_CSV = "/mnt/NAS/home/ds5725/linefinder/GT_json_canonical.csv"

OLD_SUBFOLDERS = {
    os.path.normpath(os.path.join(GT_ROOT, "GT_json_luna")),
    os.path.normpath(os.path.join(GT_ROOT, "GT_json_olivia")),
}


def normalize_text(x: Any) -> str:
    if x is None:
        return "N/A"
    s = str(x).strip()
    if s == "":
        return "N/A"
    return s


def normalize_yes_no(x: Any) -> str:
    s = normalize_text(x).lower()
    if s in {"yes", "true", "1"}:
        return "yes"
    if s in {"no", "false", "0"}:
        return "no"
    return normalize_text(x)


def normalize_facing_direction(x: Any) -> str:
    s = normalize_text(x).strip().lower()

    mapping = {
        "away": "away",
        "facing away": "away",
        "towards": "towards",
        "facing towards": "towards",
        "sideways": "sideways",
        "facing sideways": "sideways",
        "sideways-left": "sideways",
        "sideways-right": "sideways",
        "other": "other",
    }
    return mapping.get(s, normalize_text(x).lower())


def normalize_number_of_people(x: Any) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x)

    s = str(x).strip()
    if s == "":
        return "N/A"

    try:
        return str(int(float(s)))
    except Exception:
        return s


def is_old_schema_path(json_path: str) -> bool:
    parent = os.path.normpath(os.path.dirname(json_path))
    return parent in OLD_SUBFOLDERS


def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_old_schema(obj: Dict[str, Any], json_path: str) -> Dict[str, str]:
    ann = obj.get("annotation", {})
    image_id = normalize_text(obj.get("image_id"))
    if image_id == "N/A":
        image_id = os.path.splitext(os.path.basename(json_path))[0]

    return {
        "image_id": image_id,
        "json_path": os.path.abspath(json_path),
        "source_schema": "old",
        "too_hard": normalize_yes_no(ann.get("Is this image too hard?", "no")),
        "number of people": normalize_number_of_people(ann.get("Number of people in line")),
        "line facing direction": normalize_facing_direction(ann.get("What direction is the line facing")),
        "end visible": normalize_yes_no(ann.get("Do we see the end of line?")),
        "end location": normalize_text(ann.get("If yes, where is the end of the line in the image?")),
        "end direction to turn": normalize_text(
            ann.get("If not, in what direction should you move camera to see the end of the line?")
        ),
        "end person description": normalize_text(ann.get("Description of the person at the end of the line")),
        "start visible": normalize_yes_no(ann.get("Do we see the start of line?")),
        "start location": normalize_text(ann.get("If yes, where is the start of the line in the image?")),
        "start direction to turn": normalize_text(
            ann.get("If not, in what direction should you move camera to see the start of the line?")
        ),
        "start person description": normalize_text(ann.get("Description of the person at the start of the line")),
    }


def convert_new_schema(obj: Dict[str, Any], json_path: str) -> Dict[str, str]:
    image_id = os.path.splitext(os.path.basename(json_path))[0]

    people = obj.get("people", {})
    start_of_line = obj.get("start_of_line", {})
    end_of_line = obj.get("end_of_line", {})

    return {
        "image_id": image_id,
        "json_path": os.path.abspath(json_path),
        "source_schema": "new",
        "too_hard": "no",
        "number of people": normalize_number_of_people(people.get("number_of_people")),
        "line facing direction": normalize_facing_direction(people.get("direction_they_are_facing")),
        "end visible": normalize_yes_no(end_of_line.get("visible")),
        "end location": normalize_text(end_of_line.get("location_if_visible")),
        "end direction to turn": normalize_text(end_of_line.get("direction_to_turn_if_not_visible")),
        "end person description": normalize_text(people.get("end_person_description")),
        "start visible": normalize_yes_no(start_of_line.get("visible")),
        "start location": normalize_text(start_of_line.get("location_if_visible")),
        "start direction to turn": normalize_text(start_of_line.get("direction_to_turn_if_not_visible")),
        "start person description": normalize_text(people.get("start_person_description")),
    }


def iter_all_json_files(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def main():
    json_paths = iter_all_json_files(GT_ROOT)

    rows = []
    num_old = 0
    num_new = 0
    num_errors = 0

    for json_path in json_paths:
        try:
            obj = load_json(json_path)

            if is_old_schema_path(json_path):
                row = convert_old_schema(obj, json_path)
                num_old += 1
            else:
                row = convert_new_schema(obj, json_path)
                num_new += 1

            rows.append(row)

        except Exception as e:
            num_errors += 1
            print(f"[ERROR] Failed to process {json_path}: {e}")

    fieldnames = [
        "image_id",
        "json_path",
        "source_schema",
        "too_hard",
        "number of people",
        "line facing direction",
        "end visible",
        "end location",
        "end direction to turn",
        "end person description",
        "start visible",
        "start location",
        "start direction to turn",
        "start person description",
    ]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total JSON files found: {len(json_paths)}")
    print(f"Converted old-schema files: {num_old}")
    print(f"Converted new-schema files: {num_new}")
    print(f"Errors: {num_errors}")
    print(f"Output CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()