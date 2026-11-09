
import json
import os
import base64
from time import sleep
from tqdm import tqdm
from openai import OpenAI
import re

def extract_json_text(text: str) -> str:
    text = text.strip()

    # Case 1: fenced JSON block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Case 2: unfenced JSON object embedded in text
    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()

    return text
    
# =========================
# Kimi (Moonshot) client
# =========================
client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.ai/v1",
)


def image_data_url(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def _build_annotation_prompt():
    return (
        "You are an expert at analyzing a single image of a line of people. "
        "Answer the following questions STRICTLY as a single JSON object. "
        "Use the exact keys provided. For multiple-choice fields, choose ONE "
        "of the listed options verbatim. If something is not visible, pick the "
        "most appropriate option, usually 'N/A' when that option is available.\n\n"
        "Return ONLY JSON. No prose.\n\n"
        "Definitions:\n"
        "- Start of the line: the front of the line, usually the person closest "
        "to the counter or service point.\n"
        "- End of the line: the back of the line, usually the last person to join the line.\n\n"
        "Fields:\n"
        '1) "Number of people in line": integer\n'
        '2) "What direction is the line facing": one of ["towards", "away", "sideways-left", "sideways-right"]\n'
        '3) "Do we see the end of line?": one of ["yes", "no"]\n'
        '4) "If yes, where is the end of the line in the image?": one of ["far left", "center left", "center", "center right", "far right", "N/A"]\n'
        '5) "If not, in what direction should you move camera to see the end of the line?": one of ["left", "right", "back", "N/A"]\n'
        '6) "Description of the person at the end of the line": string (brief), empty if not visible\n'
        '7) "Do we see the start of line?": one of ["yes", "no"]\n'
        '8) "If yes, where is the start of the line in the image?": one of ["far left", "center left", "center", "center right", "far right", "N/A"]\n'
        '9) "If not, in what direction should you move camera to see the start of the line?": one of ["left", "right", "back", "N/A"]\n'
        '10) "Description of the person at the start of the line": string (brief), empty if not visible\n\n'
        "Return valid JSON only."
    )


def generate_reranking(image_paths, res_file_name, temperature=0.2):
    prompt = _build_annotation_prompt()

    with open(res_file_name, "a", encoding="utf-8") as f:
        for img_path in tqdm(image_paths):
            basename = os.path.basename(img_path)
            # print(img_path)

            try:
                img_url = image_data_url(img_path)

                response = client.chat.completions.create(
                    model="kimi-k2.5",
                    messages=[
                        {
                            "role": "system",
                            "content": "Return ONLY valid JSON. No extra text.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": img_url},
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        },
                    ],
                    extra_body={
                        "thinking": {"type": "disabled"}
                    },  # Pass additional request body via extra_body parameter to disable thinking
                )

                content = response.choices[0].message.content.strip()

                try:
                    parsed = json.loads(extract_json_text(content))
                    content = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    content = json.dumps(
                        {
                            "error": "invalid_json",
                            "raw_response": content,
                        },
                        ensure_ascii=False,
                    )

            except Exception as e:
                content = json.dumps({"error": str(e)}, ensure_ascii=False)

            f.write(basename + "\n" + content + "\n\n")
            f.flush()
            sleep(0.5)


# =========================
# Load images
# =========================
root = "/mnt/NAS/home/ds5725/linefinder/Images"
# subfolders = ["QueuesOutdoors","QueuesInSupermarketNew","QueuesInThemeParks"]
subfolders = ["QueuesInAirports","QueuesInSupermarkets", "QueuesInBanks", "ImagesOnline"]

all_files = []
for sub in subfolders:
    folder_path = os.path.join(root, sub)
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            all_files.append(os.path.abspath(os.path.join(dirpath, fname)))

all_files.sort()

generate_reranking(all_files, "kimi_line_old.jsonl")