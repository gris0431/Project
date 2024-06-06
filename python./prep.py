import cv2
import os
import time
import torch
import json
from torchvision import transforms

if 'model' not in globals():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

processed_files_log = "processed_files.json"

if os.path.exists(processed_files_log):
    with open(processed_files_log, 'r') as f:
        processed_files = json.load(f)
else:
    processed_files = {}

while True:
    # Get current time
    current_time = time.time()

    os.chdir('photo')

    # Variable for the output folder to store cropped silhouettes
    output_folder = "photo_network"
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the current directory
    for filename in os.listdir():
        if filename.endswith(".jpg"):
            file_creation_time = os.path.getmtime(filename)  # Get the file creation time

            # Check if the file has already been processed
            if filename in processed_files and processed_files[filename] >= file_creation_time:
                continue

            image = cv2.imread(filename)
            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform person detection
            results = model(image_rgb)
            predictions = results.pred[0]

            # Filter detections to only include people (class 0 in COCO dataset)
            people_detections = [det for det in predictions if int(det[-1]) == 0]

            if len(people_detections) > 0:
                for i, det in enumerate(people_detections):
                    x1, y1, x2, y2, conf, cls = det

                    # Convert coordinates to integers
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    # Crop silhouette from image
                    silhouette_image = image[y1:y2, x1:x2]

                    # Save cropped silhouette images in the output folder
                    cv2.imwrite(os.path.join(output_folder, f"photo_{i + 1}_{filename}"), silhouette_image)
                    print(f"Saved cropped silhouette {i + 1} from {filename}")
            else:
                print(f"No people found in {filename}")

            # Update the log with the new processing time
            processed_files[filename] = current_time

    # Save the log of processed files
    with open(processed_files_log, 'w') as f:
        json.dump(processed_files, f)

    os.chdir('..')  # Move back to the parent directory

    # Wait for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()