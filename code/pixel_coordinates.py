import cv2
import os

#  Image path
image_path = "/Users/Jamila/Desktop/LineFinder/Images/ImagesByUs/image_001.jpg"

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: The file at path '{image_path}' does not exist.")
    exit()

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Ensure it's read as a color image

# Verify if the image was loaded correctly
if image is None:
    print("Error: Could not load image. Check the file path or file format.")
    exit()
else:
    print("Image loaded successfully with shape:", image.shape)

# Function to capture mouse click events and print coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Detect left mouse button click
        print(f"Clicked at X: {x}, Y: {y}")

# Create a window and set mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', get_coordinates)

# Display the image in a loop until exiting
while True:
    cv2.imshow('image', image)
    if cv2.waitKey(20) & 0xFF == 27:  # 27 - Exit when ESC key is pressed
        break

cv2.destroyAllWindows()
