import json
import os
import base64
from time import sleep
from tqdm import tqdm
import openai

# Prefer environment variable; remove hardcoded keys for safety.
openai.api_key = os.environ["OPENAI_API_KEY"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _build_annotation_prompt():
    """
    Returns a compact, deterministic prompt with the exact questions and options
    used by the GUI tool, and asks for strict JSON output.
    """
    # Questions and options mirrored from gui1.py (keep these in lockstep). :contentReference[oaicite:2]{index=2}
    return (
        "You are an expert at analyzing a single image of a line of people. "
        "Answer the following 17 questions STRICTLY as a single JSON object. "
        "Use the exact keys provided, and for multiple-choice fields choose ONE "
        "of the listed options verbatim. If something is not visible, pick the most appropriate option (e.g., 'N/A').\n\n"
        "Return ONLY JSON. No prose.\n\n"
        "Definitions:\n"
        "- **Start of the line (front):** the person closest to the counter or service point. "
        "This is usually the direction the line is facing towards.\n"
        "- **End of the line (back):** the person farthest from the counter or service point. "
        "This is usually the last person to join the line.\n\n"
        "Fields:\n"
        '1) "number_of_turns": integer\n'
        '2) "line_shape": one of ["Straight","Curved","S-shaped","Angled","other"]\n'
        '3) "line_facing_direction": one of ["Facing towards","Facing away","Facing sideways","other"]\n'
        '4) "number_of_people_in_line": integer\n'
        '5) "line_purpose": string (short phrase)\n'
        '6) "start_person_description": string (brief)\n'
        '7) "end_person_description": string (brief)\n'
        '8) "counter_person_description": string (brief)\n'
        '9) "boundary_present": one of ["yes","no"]\n'
        '10) "boundary_types": one of ["none","cones","rope dividers","stanchions","other"]\n'
        '11) "end_of_line_visible": one of ["yes","no"]\n'
        '12) "end_of_line_location_if_visible": one of ["far left","center left","center","center right","far right","N/A"]\n'
        '13) "direction_to_turn_to_see_end_if_not_visible": one of ["left","right","back","N/A"]\n'
        '14) "start_of_line_visible": one of ["yes","no"]\n'
        '15) "start_of_line_location_if_visible": one of ["far left","center left","center","center right","far right","N/A"]\n'
        '16) "direction_to_turn_to_see_start_if_not_visible": one of ["left","right","back","N/A"]\n'
        '17) "line_completeness": one of ["full","partial"]\n\n'
        "JSON schema example (values are placeholders):\n"
        "{\n"
        '  "number_of_turns": 0,\n'
        '  "line_shape": "Straight",\n'
        '  "line_facing_direction": "Facing towards",\n'
        '  "number_of_people_in_line": 16,\n'
        '  "line_purpose": "airport",\n'
        '  "start_person_description": "man wearing hat and blue shirt",\n'
        '  "end_person_description": "person wearing black t-shirt",\n'
        '  "counter_person_description": "unknown",\n'
        '  "boundary_present": "true",\n'
        '  "boundary_types": "rope dividers",\n'
        '  "end_of_line_visible": "yes",\n'
        '  "end_of_line_location_if_visible": "far left",\n'
        '  "direction_to_turn_to_see_end_if_not_visible": "N/A",\n'
        '  "start_of_line_visible": "no",\n'
        '  "start_of_line_location_if_visible": "N/A",\n'
        '  "direction_to_turn_to_see_start_if_not_visible": "right",\n'
        '  "line_completeness": "partial"\n'
        "}"
    )

def generate_reranking(image_paths, res_file_name, temperature=0.2):
    """
    New behavior: given a list of single-image paths, ask the 17 GUI questions
    for each image and write a JSON answer per image to `res_file_name`.

    Args:
        image_paths (List[str]): list of absolute or relative image paths.
        res_file_name (str): output text file; we append one record per image:
            <basename>\n<json>\n\n
        temperature (float): sampling temperature.
    """
    prompt = _build_annotation_prompt()  # from gui spec :contentReference[oaicite:3]{index=3}

    with open(res_file_name, "a", encoding="utf-8") as f:
        for img_path in tqdm(image_paths):
            basename=os.path.basename(img_path)

            img_b64 = encode_image(img_path)
            print(img_path)
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You analyze a SINGLE image and return ONLY valid JSON "
                                    "with the specified keys and enumerated options. Do not include any extra text."
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
                max_tokens=1000,
                temperature=temperature,
            )

            content = response.choices[0].message.content.strip()
            # Optional: validate JSON quickly; if it fails, still write raw for debugging.
            try:
                parsed = json.loads(content)
                content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass  # leave as-is for troubleshooting

            f.write(os.path.basename(img_path) + "\n" + content + "\n\n")
            sleep(0.5)

root = "/vast/ds5725/linefinder/LineFinder/Images"

# Subfolders
subfolders = ["QueuesInAirports","QueuesInSupermarkets", "QueuesInBanks", "ImagesOnline"]

# Collect all absolute file paths
all_files = []
for sub in subfolders:
    folder_path = os.path.join(root, sub)
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            all_files.append(os.path.abspath(os.path.join(dirpath, f)))
all_files.sort()
# print(all_files[0])
# test=[all_files[0],all_files[1]]
generate_reranking(all_files[63:], "gpt_line_test.jsonl")