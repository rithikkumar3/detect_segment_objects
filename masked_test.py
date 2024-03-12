import random
import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO


random.seed(42)

# Initialize YOLO model
model = YOLO("models/yolov8x-seg.pt")

# Define the base directory where 'rect_left' and 'rect_right' folders are located
base_dir = 'frames'
# Define the directory to save the annotated images
ann_dir = 'ann'

# Create ann directory if it does not exist
if not os.path.exists(ann_dir):
    os.makedirs(ann_dir)

# Define the folders to process
folders = ['left', 'right']

# Set the confidence threshold
conf = 0.5

# Pre-define colors for each class
class_colors = {
    "person": (255, 0, 0),  # Red
    "bottle": (0, 255, 0),  # Green
    # "tie": (0, 0, 255)    # Blue (we'll ignore tie detections)
}

# Define the classes to ignore (e.g., 'tie')
ignored_classes = ['tie']

# Process images in both folders
for folder in folders:
    current_folder_path = os.path.join(base_dir, folder)
    # Create a corresponding directory in 'ann' to save processed images
    save_dir = os.path.join(ann_dir, folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get all image paths
    image_paths = glob.glob(os.path.join(current_folder_path, '*.png'))
    
    for image_path in image_paths:
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Initialize a black background with the same size as the original image
        background = np.zeros((height, width, 3), np.uint8)

        # Predict segmentation
        results = model.predict(img, conf=conf)

        # Create a mask image
        mask_img = background.copy()
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Skip coloring if the class is to be ignored
                if class_name in ignored_classes:
                    continue

                # Retrieve the consistent color for the class
                chosen_color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not found
                
                # Draw the mask on the image using the chosen color
                points = np.int32([mask])
                cv2.fillPoly(mask_img, points, chosen_color)
        
        # Save the result
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, mask_img)
        print(f"Processed and saved: {save_path}")

# No need for cv2.imshow or cv2.waitKey in batch processing
cv2.destroyAllWindows()