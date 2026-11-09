import os
import json
from time import sleep
from tqdm import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel

# =========================
# Gemini setup
# =========================
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_ID = "gemini-3-flash-preview"

# =========================
# Paths
# =========================
PRED_JSONL = "gemini_line_old.jsonl"
GT_DIR = "/mnt/NAS/home/ds5725/linefinder/GT_json"

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
- Return true only if the two descriptions are likely describing the same person.
- Return false if there are clear contradictions in clothing, shoes, hair, accessories, age, gender presentation, carried objects, or other identity details.
- Be conservative: if uncertain, return false.
- Missing details in one description are allowed, but contradictions should result in false.

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
            temperature=0.0,
        ),
    )
    parsed = json.loads(response.text)
    return parsed["same_person"]


# =========================
# Parsing helpers
# =========================
def load_prediction_jsonl(path):
    """
    File format:
    IMG_5954.JPG
    {...json...}

    IMG_5955.JPG
    {...json...}
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        image_name = lines[i].strip()
        i += 1

        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            print(f"Warning: missing json for {image_name}")
            break

        json_line = lines[i].strip()
        i += 1

        try:
            pred = json.loads(json_line)
        except Exception as e:
            print(f"Warning: invalid json for {image_name}: {e}")
            pred = {"error": "invalid_json", "raw": json_line}

        entries.append((image_name, pred))

    return entries


GT_CACHE = None

def build_gt_cache():
    cache = {}
    for dirpath, _, filenames in os.walk(GT_DIR):
        for f in filenames:
            if f.endswith(".json"):
                stem = os.path.splitext(f)[0]
                full_path = os.path.join(dirpath, f)
                cache[stem] = full_path
    return cache


def get_gt_path(image_name):
    global GT_CACHE
    if GT_CACHE is None:
        GT_CACHE = build_gt_cache()

    stem = os.path.splitext(os.path.basename(image_name))[0]
    return GT_CACHE.get(stem, None)


def safe_get_pred_desc(pred_dict, which):
    if not isinstance(pred_dict, dict):
        return ""

    if which == "start":
        key = "start_person_description"
    else:
        key = "end_person_description"

    val = pred_dict.get(key, "")
    if val is None:
        return ""
    return str(val).strip()


def safe_get_gt_desc(gt_dict, which):
    people = gt_dict.get("people", {})
    if which == "start":
        val = people.get("start_person_description", "")
    else:
        val = people.get("end_person_description", "")

    if val is None:
        return ""
    return str(val).strip()


def is_missing_description(desc):
    if not desc:
        return True
    desc_l = desc.lower().strip()
    return desc_l in {"n/a", "na", "none", "unknown", ""}


# =========================
# Evaluation
# =========================
def evaluate_all():
    entries = load_prediction_jsonl(PRED_JSONL)

    total_images = 0
    gt_found_images = 0

    start_total = 0
    end_total = 0
    start_true = 0
    end_true = 0
    start_gemini_compared = 0
    end_gemini_compared = 0

    for image_name, pred in tqdm(entries):
        total_images += 1
        gt_path = get_gt_path(image_name)

        if gt_path is None:
            continue

        gt_found_images += 1

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
        except Exception as e:
            print(f"GT load error for {image_name}: {e}")
            continue

        pred_start = safe_get_pred_desc(pred, "start")
        gt_start = safe_get_gt_desc(gt, "start")
        pred_end = safe_get_pred_desc(pred, "end")
        gt_end = safe_get_gt_desc(gt, "end")

                # START
        start_total += 1
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
                print(f"Start compare error for {image_name}: {e}")
            sleep(0.2)

        # END
        end_total += 1
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
                print(f"End compare error for {image_name}: {e}")
            sleep(0.2)

    print("\n===== Summary =====")
    print(f"Total images in prediction file: {total_images}")
    print(f"Images with GT found: {gt_found_images}")

    print(f"Start total evaluated: {start_total}")
    print(f"Start Gemini-compared: {start_gemini_compared}")
    if start_total > 0:
        print(f"Start accuracy: {100.0 * start_true / start_total:.2f}%")
    else:
        print("Start accuracy: N/A")

    print(f"End total evaluated: {end_total}")
    print(f"End Gemini-compared: {end_gemini_compared}")
    if end_total > 0:
        print(f"End accuracy: {100.0 * end_true / end_total:.2f}%")
    else:
        print("End accuracy: N/A")

    overall_total = start_total + end_total
    overall_true = start_true + end_true
    if overall_total > 0:
        print(f"Overall accuracy: {100.0 * overall_true / overall_total:.2f}%")
    else:
        print("Overall accuracy: N/A")


if __name__ == "__main__":
    evaluate_all()