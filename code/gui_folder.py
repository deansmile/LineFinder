import cv2
import tkinter as tk
from tkinter import ttk
import os
import json

# === CONFIGURATION ===
image_folder = "ImagesOnline"   # <- Change this to your image folder path
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# === IMAGE LIST ===
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
image_files.sort()  # Optional: sort the images

output_folder = "GT_json"
os.makedirs(output_folder, exist_ok=True)

# === MAIN LOOP ===
for image_path in image_files:
    image_base = os.path.splitext(os.path.basename(image_path))[0]
    json_output_path = os.path.join(output_folder, f"{image_base}.json")

    # Skip if already annotated
    if os.path.exists(json_output_path):
        print(f"Skipping {image_path} (annotation exists)")
        continue

    print(f"\n=== Processing: {image_path} ===")

    # Reset global variables for each image
    start_point = None
    end_point = None
    drawing = False

    # Load the image
    image = cv2.imread(image_path)
    clone = image.copy()
    original_height, original_width = image.shape[:2]

    # Prepare annotation dictionary
    annotation = {
        "line_shape": {
            "turns": None,
            "description": None
        },
        "people": {
            "number_of_people": None,
            "direction_they_are_facing": None,
            "start_person_description": None,
            "end_person_description": None,
            "counter_person_description": None
        },
        "boundary": {
            "boundary_present": None,
            "boundary_types": None
        },
        "start_of_line": {
            "visible": None,
            "location_if_visible": None,
            "direction_to_turn_if_not_visible": None
        },
        "end_of_line": {
            "visible": None,
            "location_if_visible": None,
            "direction_to_turn_if_not_visible": None
        },
        "line_purpose": None,
        "line_completeness": None,
        "line_coordinates": None  # Set in draw_line()
    }

    # === Define Functions ===

    def submit_annotations(entries):
        annotation["line_shape"]["turns"] = int(entries["Number of turns"].get())
        annotation["line_shape"]["description"] = entries["Line shape"].get()

        annotation["people"]["number_of_people"] = int(entries["Number of people in line"].get())
        annotation["people"]["direction_they_are_facing"] = entries["What direction is the line facing"].get()
        annotation["people"]["start_person_description"] = entries["Description of the person at the start of the line"].get()
        annotation["people"]["end_person_description"] = entries["Description of the person at the end of the line"].get()
        annotation["people"]["counter_person_description"] = entries["Description of the person at the counter"].get()

        annotation["line_purpose"] = entries["Line purpose"].get()

        annotation["boundary"]["boundary_present"] = entries["Is there a boundary?"].get().lower() == "yes"
        annotation["boundary"]["boundary_types"] = entries["Boundary types"].get()

        annotation["end_of_line"]["visible"] = entries["Do we see the end of line?"].get()
        annotation["end_of_line"]["location_if_visible"] = entries["If yes, where is the end of the line in the image?"].get()
        annotation["end_of_line"]["direction_to_turn_if_not_visible"] = entries["If not, in what direction should you move camera to see the end of the line?"].get()

        annotation["start_of_line"]["visible"] = entries["Do we see the start of line?"].get()
        annotation["start_of_line"]["location_if_visible"] = entries["If yes, where is the start of the line in the image?"].get()
        annotation["start_of_line"]["direction_to_turn_if_not_visible"] = entries["If not, in what direction should you move camera to see the start of the line?"].get()

        annotation["line_completeness"] = entries["Does the image show a full line or partial line?"].get()

        print("Annotation Properties:", annotation)

        image_base = os.path.splitext(os.path.basename(image_path))[0]
        json_output_path = os.path.join(output_folder, f"{image_base}.json")
        with open(json_output_path, "w") as f:
            json.dump(annotation, f, indent=4)
        print(f"Saved annotation to {image_base}.json")

        # Close GUI when submit is clicked
        root.destroy()

    def setup_annotation_gui():
        global root
        root = tk.Tk()
        root.title(f"Annotation Input: {os.path.basename(image_path)}")
        root.resizable(True, True)
        fields = {
            "Number of turns": tk.Entry,
            "Line shape": ttk.Combobox,
            "What direction is the line facing": ttk.Combobox,
            "Number of people in line": tk.Entry,
            "Line purpose": tk.Entry,
            "Description of the person at the start of the line": tk.Entry,
            "Description of the person at the end of the line": tk.Entry,
            "Description of the person at the counter": tk.Entry,
            "Is there a boundary?": ttk.Combobox,
            "Boundary types": ttk.Combobox,
            "Do we see the end of line?": ttk.Combobox,
            "If yes, where is the end of the line in the image?": ttk.Combobox,
            "If not, in what direction should you move camera to see the end of the line?": ttk.Combobox,
            "Do we see the start of line?": ttk.Combobox,
            "If yes, where is the start of the line in the image?": ttk.Combobox,
            "If not, in what direction should you move camera to see the start of the line?": ttk.Combobox,
            "Does the image show a full line or partial line?": ttk.Combobox
        }
        options = {
            "Line shape": ["Straight", "Curved", "S-shaped", "Angled","other"],
            "What direction is the line facing": ["Facing towards","Facing away","Facing sideways","other"],
            "Is there a boundary?": ["yes", "no"],
            "Boundary types": ["none", "cones", "rope dividers", "stanchions","other"],
            "Do we see the end of line?": ["yes", "no"],
            "If yes, where is the end of the line in the image?": ["far left", "center left", "center", "center right", "far right","N/A"],
            "If not, in what direction should you move camera to see the end of the line?": ["left", "right","back","N/A"],
            "Do we see the start of line?": ["yes", "no"],
            "If yes, where is the start of the line in the image?": ["far left", "center left", "center", "center right", "far right","N/A"],
            "If not, in what direction should you move camera to see the start of the line?": ["left", "right","back","N/A"],
            "Does the image show a full line or partial line?": ["full", "partial"]
        }
        entries = {}
        for field, widget_type in fields.items():
            row = tk.Frame(root)
            label = tk.Label(row, width=70, text=field, anchor='w')
            if field in options:
                entry = widget_type(row, values=options[field], state="readonly")
                entry.set(options[field][0])
            else:
                entry = widget_type(row)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            entries[field] = entry
        submit_button = tk.Button(root, text="Submit", command=lambda: submit_annotations(entries))
        submit_button.pack(side=tk.BOTTOM, pady=10)
        root.update()
        return root

    def draw_line(event, x, y, flags, param):
        global start_point, end_point, drawing, image
        if event == cv2.EVENT_LBUTTONDOWN:
            if start_point is None:
                start_point = (x, y)
                cv2.circle(image, start_point, 5, (255, 0, 0), -1)  # Blue dot
                cv2.imshow("Annotation Tool", image)
            else:
                end_point = (x, y)
                cv2.arrowedLine(image, start_point, end_point, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA, tipLength=0.05)
                cv2.imshow("Annotation Tool", image)

                # Save line coordinates
                annotation["line_coordinates"] = {
                    "start": {"x": start_point[0], "y": start_point[1]},
                    "end": {"x": end_point[0], "y": end_point[1]}
                }
                print(f"Line coordinates saved: Start - {start_point}, End - {end_point}")

    # === Setup drawing window (image size)
    cv2.namedWindow("Annotation Tool", cv2.WINDOW_NORMAL)
    cv2.imshow("Annotation Tool", image)
    cv2.setMouseCallback("Annotation Tool", draw_line)

    # === Setup annotation GUI
    gui = setup_annotation_gui()

    # === Start GUI loop
    gui.mainloop()

    # === Cleanup OpenCV window
    cv2.destroyAllWindows()

print("\nAll images processed!")
