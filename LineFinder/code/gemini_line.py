import os
try:  # load API keys from the repo-root .env (python-dotenv optional)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
import json
import os
from time import sleep
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel

# =========================
# Gemini setup
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_ID = "gemini-3-flash-preview"  # or newer if available


# =========================
# JSON schema (STRICT)
# =========================
class QueueAnswer(BaseModel):
    number_of_people: int
    line_direction: str
    end_visible: str
    end_location: str
    end_camera_direction: str
    end_person_description: str
    start_visible: str
    start_location: str
    start_camera_direction: str
    start_person_description: str


# =========================
# Prompt (same as GPT version)
# =========================
def build_prompt():
    return (
        "You are an expert at analyzing a single image of a line of people.\n"
        "Return STRICT JSON only.\n\n"

        "Fields:\n"
        'number_of_people: integer\n'
        'line_direction: ["towards", "away", "sideways-left", "sideways-right"]\n'
        'end_visible: ["yes","no"]\n'
        'end_location: ["far left","center left","center","center right","far right","N/A"]\n'
        'end_camera_direction: ["left","right","back","N/A"]\n'
        'end_person_description: string\n'
        'start_visible: ["yes","no"]\n'
        'start_location: ["far left","center left","center","center right","far right","N/A"]\n'
        'start_camera_direction: ["left","right","back","N/A"]\n'
        'start_person_description: string\n'
    )


# =========================
# Single image inference
# =========================
def analyze_image(img_path):
    image = Image.open(img_path)
    image.thumbnail([512, 512])  # Gemini requirement

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            "Return ONLY JSON.",
            build_prompt(),
            image
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QueueAnswer,
            temperature=0.2,
        ),
    )

    return response.text


# =========================
# Batch processing
# =========================
def generate_reranking(image_paths, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        for img_path in tqdm(image_paths):
            basename = os.path.basename(img_path)

            try:
                result = analyze_image(img_path)

                try:
                    parsed = json.loads(result)
                except:
                    parsed = {"error": "invalid_json", "raw": result}

                f.write(basename + "\n")
                f.write(json.dumps(parsed) + "\n\n")
                f.flush()

            except Exception as e:
                print(f"Error: {img_path} -> {e}")
                f.write(basename + "\n")
                f.write(json.dumps({"error": str(e)}) + "\n\n")

            sleep(0.2)


# =========================
# Image sets
# =========================
IMAGE_SETS = {
    # Original collection (airports, supermarkets, banks, online images)
    "old": ["QueuesInAirports", "QueuesInSupermarkets", "QueuesInBanks", "ImagesOnline"],
    # Luna/Olivia collection
    "new": ["QueuesOutdoors", "QueuesInSupermarketNew", "QueuesInThemeParks"],
}
IMAGE_SETS["full"] = IMAGE_SETS["old"] + IMAGE_SETS["new"]

DEFAULT_OUTPUTS = {
    "old": "gemini_line_old.jsonl",
    "new": "gemini_line_luna_olivia.jsonl",
    "full": "gemini_line_full.jsonl",
}


def collect_images(root, subfolders):
    all_files = []
    for sub in subfolders:
        folder_path = os.path.join(root, sub)
        for dirpath, _, filenames in os.walk(folder_path):
            for f in filenames:
                all_files.append(os.path.abspath(os.path.join(dirpath, f)))
    all_files.sort()
    return all_files


if __name__ == "__main__":
    import argparse
    from lf_paths import lf_path

    ap = argparse.ArgumentParser(description="Ask Gemini the LineFinder questions for a set of images.")
    ap.add_argument("--image_set", required=True, choices=sorted(IMAGE_SETS),
                    help="old = airports/supermarkets/banks/online; new = outdoors/supermarket-new/theme parks; full = both")
    ap.add_argument("--output", default=None,
                    help="Output JSONL (default depends on --image_set)")
    args = ap.parse_args()

    files = collect_images(lf_path("Images"), IMAGE_SETS[args.image_set])
    out = args.output or DEFAULT_OUTPUTS[args.image_set]
    print(f"{args.image_set}: {len(files)} images -> {out}")
    generate_reranking(files, out)
