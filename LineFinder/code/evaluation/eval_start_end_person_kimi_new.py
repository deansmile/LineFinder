import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
import json
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
PRED_JSONL = "kimi_line_luna_olivia.jsonl"
GT_DIR = lf_path("GT_json")  # change if needed


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
def is_image_line(line):
    s = line.strip().lower()
    return s.endswith((".jpg", ".jpeg", ".png", ".webp"))


def load_prediction_jsonl(path):
    """
    Robust loader for format like:

    supermarket_001.jpg
    {"Number of people in line": ..., ...}

    or multi-line JSON blocks.
    """
    entries = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    n = len(lines)

    while i < n:
        while i < n and not lines[i].strip():
            i += 1

        if i >= n:
            break

        if not is_image_line(lines[i]):
            print(f"Warning: expected image filename, got: {lines[i]}")
            i += 1
            continue

        image_name = lines[i].strip()
        i += 1

        while i < n and not lines[i].strip():
            i += 1

        if i >= n:
            print(f"Warning: missing json for {image_name}")
            break

        json_lines = []
        brace_balance = 0
        started = False

        while i < n:
            line = lines[i]

            if is_image_line(line) and started and brace_balance <= 0:
                break

            if line.strip():
                json_lines.append(line)
                started = True
                brace_balance += line.count("{") - line.count("}")

            i += 1

            if started and brace_balance <= 0:
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j >= n or is_image_line(lines[j]):
                    break

        raw_json = "\n".join(json_lines).strip()

        try:
            pred = json.loads(raw_json)
        except Exception as e:
            print(f"Warning: invalid json for {image_name}: {e}")
            print(f"Raw JSON block:\n{raw_json}\n")
            pred = {"error": "invalid_json", "raw": raw_json}

        entries.append((image_name, pred))

    return entries


GT_CACHE = None

def build_gt_cache():
    cache = {}
    for dirpath, _, filenames in os.walk(GT_DIR):
        for f in filenames:
            if f.endswith(".json"):
                stem = os.path.splitext(f)[0]
                cache[stem] = os.path.join(dirpath, f)
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
        key = "Description of the person at the start of the line"
    else:
        key = "Description of the person at the end of the line"

    val = pred_dict.get(key, "")
    if val is None:
        return ""
    return str(val).strip()


def safe_get_gt_desc(gt_dict, which):
    annotation = gt_dict.get("annotation", {})
    if which == "start":
        key = "Description of the person at the start of the line"
    else:
        key = "Description of the person at the end of the line"

    val = annotation.get(key, "")
    if val is None:
        return ""
    return str(val).strip()


def is_too_hard(gt_dict):
    annotation = gt_dict.get("annotation", {})
    val = str(annotation.get("Is this image too hard?", "")).strip().lower()
    return val == "yes"


def is_missing_description(desc):
    if not desc:
        return True
    desc_l = desc.lower().strip()
    return desc_l in {"", "n/a", "na", "none", "unknown"}


# =========================
# Evaluation
# =========================
def evaluate_all():
    entries = load_prediction_jsonl(PRED_JSONL)

    total_images_in_pred = 0
    gt_found_images = 0
    not_too_hard_images = 0

    start_total = 0
    end_total = 0
    start_true = 0
    end_true = 0
    start_gemini_compared = 0
    end_gemini_compared = 0

    for image_name, pred in tqdm(entries):
        total_images_in_pred += 1
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

        if is_too_hard(gt):
            continue

        not_too_hard_images += 1

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

    overall_total = start_total + end_total
    overall_true = start_true + end_true

    print("\n===== Summary =====")
    print(f"Total images in prediction file: {total_images_in_pred}")
    print(f"Images with GT found: {gt_found_images}")
    print(f"Not-too-hard images evaluated: {not_too_hard_images}")

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

    if overall_total > 0:
        print(f"Overall accuracy: {100.0 * overall_true / overall_total:.2f}%")
    else:
        print("Overall accuracy: N/A")


if __name__ == "__main__":
    evaluate_all()