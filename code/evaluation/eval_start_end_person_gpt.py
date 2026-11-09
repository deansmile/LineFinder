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
PRED_JSONL = "gpt_line_old.jsonl"   # change if needed
GT_DIR = "/mnt/NAS/home/ds5725/linefinder/GT_json"


class SamePersonAnswer(BaseModel):
    same_person: bool


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


def strip_code_fences(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def load_prediction_jsonl(path):
    entries = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    n = len(lines)

    while i < n:
        if not lines[i].strip():
            i += 1
            continue

        image_name = lines[i].strip()
        i += 1

        while i < n and not lines[i].strip():
            i += 1

        if i >= n:
            print(f"Warning: missing json block for {image_name}")
            break

        block_lines = []

        if lines[i].strip().startswith("```"):
            block_lines.append(lines[i])
            i += 1
            while i < n:
                block_lines.append(lines[i])
                if lines[i].strip() == "```":
                    i += 1
                    break
                i += 1
        else:
            brace_balance = 0
            started = False
            while i < n:
                cur = lines[i]
                if not started and not cur.strip():
                    i += 1
                    continue
                if not started:
                    started = True
                block_lines.append(cur)
                brace_balance += cur.count("{") - cur.count("}")
                i += 1
                if started and brace_balance <= 0 and "{" in "\n".join(block_lines):
                    break

        raw_block = "\n".join(block_lines).strip()
        raw_json = strip_code_fences(raw_block)

        try:
            pred = json.loads(raw_json)
        except Exception as e:
            print(f"Warning: invalid json for {image_name}: {e}")
            pred = {"error": "invalid_json", "raw": raw_block}

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
    key = "start_person_description" if which == "start" else "end_person_description"
    val = pred_dict.get(key, "")
    return "" if val is None else str(val).strip()


def safe_get_gt_desc(gt_dict, which):
    people = gt_dict.get("people", {})
    key = "start_person_description" if which == "start" else "end_person_description"
    val = people.get(key, "")
    return "" if val is None else str(val).strip()


def is_missing_description(desc):
    if not desc:
        return True
    desc_l = desc.lower().strip()
    return desc_l in {"", "n/a", "na", "none", "unknown"}


def evaluate_one_side(pred_desc, gt_desc):
    """
    Returns:
        correct: bool
        compared_with_gemini: bool
    """
    pred_missing = is_missing_description(pred_desc)
    gt_missing = is_missing_description(gt_desc)

    # both missing => correct
    if pred_missing and gt_missing:
        return True, False

    # one missing, one not => wrong
    if pred_missing != gt_missing:
        return False, False

    # both present => ask Gemini
    same = gemini_same_person(pred_desc, gt_desc)
    return same, True


def evaluate_all():
    entries = load_prediction_jsonl(PRED_JSONL)

    total_images = 0
    gt_found_images = 0

    start_total = 0
    start_correct = 0
    start_gemini_compared = 0

    end_total = 0
    end_correct = 0
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
        try:
            correct, used_gemini = evaluate_one_side(pred_start, gt_start)
            if correct:
                start_correct += 1
            if used_gemini:
                start_gemini_compared += 1
                sleep(0.2)
        except Exception as e:
            print(f"Start compare error for {image_name}: {e}")

        # END
        end_total += 1
        try:
            correct, used_gemini = evaluate_one_side(pred_end, gt_end)
            if correct:
                end_correct += 1
            if used_gemini:
                end_gemini_compared += 1
                sleep(0.2)
        except Exception as e:
            print(f"End compare error for {image_name}: {e}")

    overall_correct = start_correct + end_correct
    overall_total = start_total + end_total

    print("\n===== Summary =====")
    print(f"Total images in prediction file: {total_images}")
    print(f"Images with GT found: {gt_found_images}")

    print(f"Start total evaluated: {start_total}")
    print(f"Start Gemini-compared: {start_gemini_compared}")
    print(f"Start accuracy: {100.0 * start_correct / start_total:.2f}%" if start_total > 0 else "Start accuracy: N/A")

    print(f"End total evaluated: {end_total}")
    print(f"End Gemini-compared: {end_gemini_compared}")
    print(f"End accuracy: {100.0 * end_correct / end_total:.2f}%" if end_total > 0 else "End accuracy: N/A")

    print(f"Overall accuracy: {100.0 * overall_correct / overall_total:.2f}%" if overall_total > 0 else "Overall accuracy: N/A")


if __name__ == "__main__":
    evaluate_all()