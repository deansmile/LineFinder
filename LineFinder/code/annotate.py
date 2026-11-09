

import json

# Data to be saved in JSON format
annotation_data = {
    "image_id": "image_000.jpg",

    # Line shape details
    "line_shape": {
        "turns": 0,
        "description": "Straight line"  # Options: Curved, S-shaped, 90 degrees (angled)
    },

    "start_of_line": {
        "coordinates": [200, 400],  # in image pixels 

        # Options: Far left, center left, center, center right, far right
        "where_in_image_field_of_view": "center left",  

        "estimated_distance": 2.4  # in meters 
    },

    "end_of_line": {
        "coordinates": [200, 400],  # in image pixels 

        # Options: Far left, center left, center, center right, far right
        "where_in_image_field_of_view": "far right",

        "estimated_distance": 10,

        "person_at_the_end": "Male, blue jacket",  # Example: Gender, clothing colors

        "do_you_see_end": True,  # If False, check camera_rotation_suggestion

        "camera_rotation_suggestion": "Rotate right to view the end of the line"
    },

    "people": {
        "number_of_people": 10,

        # Options: front, center left, center, center right, far right
        "direction_they_are_facing": "center right"
    },

    "higher_level_info": {
        "line_purpose": "Theme park ride",
        "estimated_waiting_time_minutes": 30
    },

    "boundary": {
        "boundary_present": True,
        "boundary_types": ["cones", "rope dividers", "stanchions"]
    }
}

# Path to save the JSON file
file_path = '/Users/Jamila/Desktop/LineFinder/Annotations/image_000.json'  # Ensure the folder exists

# Write the data to a JSON file
with open(file_path, 'w') as file:
    json.dump(annotation_data, file, indent=4)

print("JSON file has been created and saved.")  # Confirm script execution



