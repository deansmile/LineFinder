import csv
import os

GT_PATH = "gt_facing.out"
PRED_CSV = "queue_facing_results.csv"

# -----------------------
# Load Ground Truth
# -----------------------
gt = {}  # { 'IMG_5954': 'away', ... }

with open(GT_PATH, "r") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split()
        img_id = parts[0]
        label = " ".join(parts[1:]).lower()  # "facing away"
        label = label.replace("facing ", "")  # "away"
        gt[img_id] = label

print(f"Loaded {len(gt)} GT entries")


# -----------------------
# Load Predictions
# -----------------------
pred = {}  # { 'IMG_5954': 'away', ... }

with open(PRED_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["image_path"]
        pred_label = row["facing_direction"].lower()

        # extract basename without extension
        img_id = os.path.splitext(os.path.basename(img_path))[0]

        # normalize predictions
        if pred_label.startswith("sideways"):
            pred_label = "sideways"

        pred[img_id] = pred_label

print(f"Loaded {len(pred)} predictions")


# -----------------------
# Compare
# -----------------------
correct = 0
total = 0
errors = []

for img_id, gt_label in gt.items():
    if img_id not in pred:
        print(f"[WARN] No prediction for {img_id}")
        continue

    pred_label = pred[img_id]

    if pred_label == gt_label or gt_label=="other":
        correct += 1
    else:
        errors.append((img_id, gt_label, pred_label))

    total += 1

# -----------------------
# Output Accuracy
# -----------------------
accuracy = correct / total if total > 0 else 0
print(f"\nAccuracy: {accuracy:.2%}  ({correct}/{total})")

if errors:
    print("\nIncorrect predictions:")
    for img_id, gt_label, pred_label in errors:
        print(f"  {img_id}: GT={gt_label}  |  Pred={pred_label}")
