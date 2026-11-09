from lf_paths import lf_path  # LINEFINDER_ROOT env var
import csv

CSV_PATH = lf_path("GT_json_canonical.csv")


def count_hard_images(csv_path):
    hard = 0
    not_hard = 0
    unknown = 0
    total = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += 1
            val = str(row.get("too_hard", "")).strip().lower()

            if val in {"yes", "true", "1"}:
                hard += 1
            elif val in {"no", "false", "0"}:
                not_hard += 1
            else:
                unknown += 1

    print("===== Hard Image Statistics =====")
    print(f"Total rows: {total}")
    print(f"Hard images: {hard}")
    print(f"Not hard images: {not_hard}")
    print(f"Unknown / missing: {unknown}")


if __name__ == "__main__":
    count_hard_images(CSV_PATH)