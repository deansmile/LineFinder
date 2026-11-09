from lf_paths import lf_path  # LINEFINDER_ROOT env var
import csv

def normalize_yes_no(x):
    s = str(x).strip().lower()
    if s in {"yes", "true", "1"}:
        return "yes"
    if s in {"no", "false", "0"}:
        return "no"
    return s

def is_empty_text(x):
    return str(x).strip() == ""

def check_result_csv(csv_path):
    total_rows = 0
    missing_start = []
    missing_end = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            image_id = row.get("image_id", "")
            image_path = row.get("image_path", "")
            status = str(row.get("status", "")).strip().lower()

            start_visible = normalize_yes_no(row.get("start_of_line_visible", ""))
            end_visible = normalize_yes_no(row.get("end_of_line_visible", ""))
            start_desc = row.get("start_description", "")
            end_desc = row.get("end_description", "")

            if start_visible == "yes" and is_empty_text(start_desc):
                missing_start.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "status": status,
                })

            if end_visible == "yes" and is_empty_text(end_desc):
                missing_end.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "status": status,
                })

    print(f"Total rows checked: {total_rows}")
    print(f"Missing START descriptions for visible start: {len(missing_start)}")
    print(f"Missing END descriptions for visible end: {len(missing_end)}")

    if missing_start:
        print("\nRows missing START description:")
        for item in missing_start:
            print(f"  {item['image_id']} | {item['image_path']} | status={item['status']}")

    if missing_end:
        print("\nRows missing END description:")
        for item in missing_end:
            print(f"  {item['image_id']} | {item['image_path']} | status={item['status']}")

if __name__ == "__main__":
    csv_path = lf_path("code/start_end_person_descriptions_new.csv")
    check_result_csv(csv_path)