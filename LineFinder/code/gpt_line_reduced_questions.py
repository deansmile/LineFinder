from lf_paths import lf_path  # LINEFINDER_ROOT env var
import os
try:  # load API keys from the repo-root .env (python-dotenv optional)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
import json
import os
import base64
from time import sleep
from tqdm import tqdm
import openai

# Prefer environment variable for safety, e.g.:
# export OPENAI_API_KEY="your_key_here"
openai.api_key = os.getenv("OPENAI_API_KEY")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _build_annotation_prompt():
    """
    Returns a compact, deterministic prompt with only the requested questions
    and asks for strict JSON output.
    """
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
        "JSON schema example:\n"
        "{\n"
        '  "Number of people in line": 8,\n'
        '  "What direction is the line facing": "away",\n'
        '  "Do we see the end of line?": "no",\n'
        '  "If yes, where is the end of the line in the image?": "N/A",\n'
        '  "If not, in what direction should you move camera to see the end of the line?": "left",\n'
        '  "Description of the person at the end of the line": "",\n'
        '  "Do we see the start of line?": "yes",\n'
        '  "If yes, where is the start of the line in the image?": "center right",\n'
        '  "If not, in what direction should you move camera to see the start of the line?": "N/A",\n'
        '  "Description of the person at the start of the line": "man in gray suit, black shoes, short hair, black shoulder bag"\n'
        "}"
    )


def generate_reranking(image_paths, res_file_name, temperature=0.2):
    """
    Given a list of single-image paths, ask the requested queue questions
    for each image and write a JSON answer per image to `res_file_name`.

    Args:
        image_paths (List[str]): list of absolute or relative image paths.
        res_file_name (str): output text file; we append one record per image:
            <basename>\n<json>\n\n
        temperature (float): sampling temperature.
    """
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    prompt = _build_annotation_prompt()

    with open(res_file_name, "a", encoding="utf-8") as f:
        for img_path in tqdm(image_paths):
            basename = os.path.basename(img_path)
            img_b64 = encode_image(img_path)
            print(img_path)

            response = openai.ChatCompletion.create(
                model="gpt-5.4-mini-2026-03-17",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You analyze a SINGLE image and return ONLY valid JSON "
                                    "with the specified keys and enumerated options. "
                                    "Do not include any extra text."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    },
                ],
            )

            content = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(content)
                content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass

            f.write(basename + "\n" + content + "\n\n")
            sleep(0.5)


root = lf_path("Images")

# Subfolders
subfolders = ["QueuesOutdoors","QueuesInSupermarketNew","QueuesInThemeParks"]

# Collect all absolute file paths
all_files = []
for sub in subfolders:
    folder_path = os.path.join(root, sub)
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            all_files.append(os.path.abspath(os.path.join(dirpath, f)))

all_files.sort()

# print(all_files[:1])
generate_reranking(all_files, "gpt_line_luna_olivia.jsonl")
