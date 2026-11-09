import os
import csv
import numpy as np

# -------- INPUT --------
IMAGE_LIST_TXT = "olivia_luna_image_paths.txt"
BBOX_DIR = "/scratch/ds5725/linefinder/LineFinder/bbox_orient"
OUTPUT_CSV = "OL_number_of_people.csv"


def read_image_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    image_paths = read_image_list(IMAGE_LIST_TXT)
    print(f"Loaded {len(image_paths)} images from txt")

    rows = []

    for img_path in image_paths:
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        bbox_path = os.path.join(BBOX_DIR, f"{image_id}_bboxes.npy")

        if not os.path.isfile(bbox_path):
            rows.append({
                "image_id": image_id,
                "image_path": img_path,
                "estimated_number_of_persons": "",
                "status": "missing_bbox"
            })
            continue

        try:
            bboxes = np.load(bbox_path)
            num_persons = int(bboxes.shape[0])
            status = "ok"
        except Exception as e:
            num_persons = ""
            status = f"fail:{e}"

        rows.append({
            "image_id": image_id,
            "image_path": img_path,
            "estimated_number_of_persons": num_persons,
            "status": status
        })

    # Write CSV
    cols = ["image_id", "image_path", "estimated_number_of_persons", "status"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()