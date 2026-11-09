import csv


def is_missing(val):
    if val is None:
        return True
    s = str(val).strip().lower()
    return s == "" or s in {"n/a", "na", "none", "null"}


visibility_csv = "/scratch/ds5725/linefinder/LineFinder/code/OL_line_visibility.csv"
desc_csv = "/scratch/ds5725/linefinder/LineFinder/code/start_end_person_descriptions_rerun.csv"


# -------------------------------------------------------
# Load description CSV
# -------------------------------------------------------

desc_dict = {}

with open(desc_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        desc_dict[row["image_id"].strip()] = row


# -------------------------------------------------------
# Compare
# -------------------------------------------------------

start_need = 0
start_have = 0
start_missing = 0

end_need = 0
end_have = 0
end_missing = 0


with open(visibility_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:

        image_id = row["image_id"].strip()

        start_visible = row["start_of_line_visible"].strip().lower() == "yes"
        end_visible = row["end_of_line_visible"].strip().lower() == "yes"

        desc_row = desc_dict.get(image_id)

        start_desc = ""
        end_desc = ""

        if desc_row:
            start_desc = desc_row.get("start_description", "")
            end_desc = desc_row.get("end_description", "")

        # -------------------------
        # START
        # -------------------------

        if start_visible:
            start_need += 1

            if is_missing(start_desc):
                start_missing += 1
                print(f"{image_id} -> START visible but description missing")
            else:
                start_have += 1

        # -------------------------
        # END
        # -------------------------

        if end_visible:
            end_need += 1

            if is_missing(end_desc):
                end_missing += 1
                print(f"{image_id} -> END visible but description missing")
            else:
                end_have += 1


# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print("\n====================")
print("START of line")
print("Need description:", start_need)
print("Have description:", start_have)
print("Missing description:", start_missing)

print("\nEND of line")
print("Need description:", end_need)
print("Have description:", end_have)
print("Missing description:", end_missing)