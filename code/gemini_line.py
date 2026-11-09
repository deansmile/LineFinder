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
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
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
# Load images (same as before)
# =========================
root = "/mnt/NAS/home/ds5725/linefinder/Images"
# subfolders = ["QueuesOutdoors","QueuesInSupermarketNew","QueuesInThemeParks"]
subfolders = ["QueuesInAirports","QueuesInSupermarkets", "QueuesInBanks", "ImagesOnline"]

all_files = []
for sub in subfolders:
    folder_path = os.path.join(root, sub)
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            all_files.append(os.path.abspath(os.path.join(dirpath, f)))

all_files.sort()

generate_reranking(all_files, "gemini_line_old.jsonl")