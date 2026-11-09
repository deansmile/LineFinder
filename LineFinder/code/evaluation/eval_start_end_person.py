import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
import json
import csv
from time import sleep
from tqdm import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel

# =========================
# Gemini setup for description matching
# =========================
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "gemini-3-flash-preview"

# =========================
# Paths
# =========================
PRED_CSV = lf_path("code/start_end_person_descriptions_new.csv")
GT_DIR = lf_path("GT_json")
GT_CSV = lf_path("GT_json_canonical.csv")

# =========================
# Output schema
# =========================
class SamePersonAnswer(BaseModel):
    same_person: bool


# =========================
# Gemini prompt
# =========================
def build_prompt(desc1, desc2):
    return f"""
You compare two short person descriptions and decide whether they describe the same person.

Return STRICT JSON only with this field:
same_person: boolean

Rules:
- Return true if the two descriptions are likely describing the same person.
- Return false if there are clear contradictions

Description 1: {desc1}
Description 2: {desc2}
"""


def gemini_same_person(desc1, desc2):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=build_prompt(desc1, desc2),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SamePersonAnswer,
        ),
    )
    parsed = json.loads(response.text)
    return parsed["same_person"]

def load_gt_csv(path):
    gt = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                gt[image_id] = row
    return gt
    
# =========================
# Prediction CSV loader
# =========================
def load_prediction_csv(path):
    """
    Expected columns include:
      image_id,image_path,start_of_line_visible,end_of_line_visible,
      start_description,end_description,status,error
    """
    entries = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                continue
            entries.append(row)

    return entries




def safe_get_pred_desc(pred_row, which):
    if not isinstance(pred_row, dict):
        return ""

    key = "start_description" if which == "start" else "end_description"
    val = pred_row.get(key, "")
    if val is None:
        return ""
    return str(val).strip()


def safe_get_pred_visible(pred_row, which):
    key = "start_of_line_visible" if which == "start" else "end_of_line_visible"
    val = str(pred_row.get(key, "")).strip().lower()
    return val in {"yes", "true", "1"}

def safe_get_gt_desc(gt_row, which):
    key = "start person description" if which == "start" else "end person description"
    val = gt_row.get(key, "")
    if val is None:
        return ""
    return str(val).strip()


def safe_get_gt_visible(gt_row, which):
    key = "start visible" if which == "start" else "end visible"
    val = str(gt_row.get(key, "")).strip().lower()
    return val in {"yes", "true", "1"}


def is_too_hard(gt_row):
    val = str(gt_row.get("too_hard", "")).strip().lower()
    return val in {"yes", "true", "1"}


def is_missing_description(desc):
    if not desc:
        return True
    desc_l = desc.lower().strip()
    return desc_l in {"", "n/a", "na", "none", "unknown"}


def evaluate_all():
    entries = load_prediction_csv(PRED_CSV)
    gt_dict = load_gt_csv(GT_CSV)

    total_rows_in_pred = 0
    gt_found_images = 0
    not_too_hard_images = 0

    start_total = 0
    end_total = 0
    start_true = 0
    end_true = 0
    start_gemini_compared = 0
    end_gemini_compared = 0

    for pred in tqdm(entries):
        total_rows_in_pred += 1
        image_id = str(pred.get("image_id", "")).strip()

        gt_row = gt_dict.get(image_id)
        if gt_row is None:
            continue

        gt_found_images += 1

        if is_too_hard(gt_row):
            continue

        not_too_hard_images += 1

        pred_start_visible = safe_get_pred_visible(pred, "start")
        pred_end_visible = safe_get_pred_visible(pred, "end")
        gt_start_visible = safe_get_gt_visible(gt_row, "start")
        gt_end_visible = safe_get_gt_visible(gt_row, "end")

        pred_start = safe_get_pred_desc(pred, "start")
        gt_start = safe_get_gt_desc(gt_row, "start")
        pred_end = safe_get_pred_desc(pred, "end")
        gt_end = safe_get_gt_desc(gt_row, "end")

        # =========================
        # START
        # =========================
        start_total += 1

        if (not gt_start_visible) and (not pred_start_visible):
            start_true += 1
        elif gt_start_visible != pred_start_visible:
            pass
        else:
            pred_start_missing = is_missing_description(pred_start)
            gt_start_missing = is_missing_description(gt_start)

            if pred_start_missing and gt_start_missing:
                start_true += 1
            elif pred_start_missing != gt_start_missing:
                pass
            else:
                try:
                    same = gemini_same_person(pred_start, gt_start)
                    start_gemini_compared += 1
                    if same:
                        start_true += 1
                except Exception as e:
                    print(f"Start compare error for {image_id}: {e}")
                sleep(0.2)

        # =========================
        # END
        # =========================
        end_total += 1

        if (not gt_end_visible) and (not pred_end_visible):
            end_true += 1
        elif gt_end_visible != pred_end_visible:
            pass
        else:
            pred_end_missing = is_missing_description(pred_end)
            gt_end_missing = is_missing_description(gt_end)

            if pred_end_missing and gt_end_missing:
                end_true += 1
            elif pred_end_missing != gt_end_missing:
                pass
            else:
                try:
                    same = gemini_same_person(pred_end, gt_end)
                    end_gemini_compared += 1
                    if same:
                        end_true += 1
                except Exception as e:
                    print(f"End compare error for {image_id}: {e}")
                sleep(0.2)

    overall_total = start_total + end_total
    overall_true = start_true + end_true

    print("\n===== Summary =====")
    print(f"Total rows in prediction CSV: {total_rows_in_pred}")
    print(f"Rows with GT found: {gt_found_images}")
    print(f"Not-too-hard images evaluated: {not_too_hard_images}")

    print(f"Start total evaluated: {start_total}")
    print(f"Start Gemini-compared: {start_gemini_compared}")
    if start_total > 0:
        print(f"Start accuracy: {100.0 * start_true / start_total:.2f}%")

    print(f"End total evaluated: {end_total}")
    print(f"End Gemini-compared: {end_gemini_compared}")
    if end_total > 0:
        print(f"End accuracy: {100.0 * end_true / end_total:.2f}%")

    if overall_total > 0:
        print(f"Overall accuracy: {100.0 * overall_true / overall_total:.2f}%")

if __name__ == "__main__":
    evaluate_all()