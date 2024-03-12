from ultralytics import YOLO
import random
import cv2
import numpy as np
import os
import glob

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

# # Prepare a fixed color for each class to ensure consistency across images
# yolo_classes = list(model.names.values())
# class_colors = {clas: random.choices(range(256), k=3) for clas in yolo_classes}

class_colors = {}

# Function to generate a random color
def get_color():
    return random.choices(range(256), k=3)

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
                points = np.int32([mask])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name not in class_colors:
                    class_colors[class_name] = get_color()

                # Retrieve the color for the class
                chosen_color = class_colors[class_name]
                
                # Draw the mask on the image using the chosen color
                points = np.int32([mask])
                cv2.fillPoly(mask_img, points, chosen_color)
        
        # Save the result
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, mask_img)
        print(f"Processed and saved: {save_path}")

# No need for cv2.imshow or cv2.waitKey in batch processing
cv2.destroyAllWindows()


# from ultralytics import YOLO
# import random
# import cv2
# import numpy as np
# import os
# import glob

# random.seed(42)

# # Initialize YOLO model
# model = YOLO("models/yolov8x-seg.pt")

# # Define the base directory where 'rect_left' and 'rect_right' folders are located
# base_dir = 'frames'
# # Define the directory to save the annotated images
# ann_dir = 'ann'

# # Create ann directory if it does not exist
# if not os.path.exists(ann_dir):
#     os.makedirs(ann_dir)

# # Define the folders to process
# folders = ['left', 'right']

# # Set the confidence threshold
# conf = 0.5

# # Initialize dictionary to store object counts
# object_counts = {}

# # Process images in both folders
# for folder in folders:
#     current_folder_path = os.path.join(base_dir, folder)
#     # Create a corresponding directory in 'ann' to save processed images
#     save_dir = os.path.join(ann_dir, folder)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # Get all image paths
#     image_paths = glob.glob(os.path.join(current_folder_path, '*.png'))
    
#     for image_path in image_paths:
#         img = cv2.imread(image_path)
#         height, width = img.shape[:2]

#         # Initialize a black background with the same size as the original image
#         background = np.zeros((height, width, 3), np.uint8)

#         # Predict segmentation
#         results = model.predict(img, conf=conf)

#         # Create a mask image
#         mask_img = background.copy()
        
#         # Initialize count for current image
#         total_objects = 0

#         for result in results:
#             total_objects += len(result.boxes)
#             for mask, box in zip(result.masks.xy, result.boxes):
#                 points = np.int32([mask])
#                 class_id = int(box.cls[0])
#                 class_name = model.names[class_id]
#                 chosen_color = random.choices(range(256), k=3)
#                 cv2.fillPoly(mask_img, points, chosen_color)
        
#         # Save the result
#         save_path = os.path.join(save_dir, os.path.basename(image_path))
#         cv2.imwrite(save_path, mask_img)
#         print(f"Processed and saved: {save_path}")
        
#         # Update the dictionary to count images with specific number of objects
#         if total_objects not in object_counts:
#             object_counts[total_objects] = 1
#         else:
#             object_counts[total_objects] += 1

# # Print the counts of images with specific number of objects
# print("Number of images with specific number of objects:")
# for count, num_images in object_counts.items():
#     print(f"{num_images} images have {count} object(s)")

# # No need for cv2.imshow or cv2.waitKey in batch processing
# cv2.destroyAllWindows()
