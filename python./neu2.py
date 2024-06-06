import os
import json
import time
import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def extract_unique_labels(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df['label'].unique()


def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def classify_cropped_images(model, output, target_size, cropped_img_path, results_file, class_ids_file):
    preprocessed_img = preprocess_image(cropped_img_path, target_size)
    prediction = model.predict(preprocessed_img)
    confidence_score = np.max(prediction)  # Получаем точность предсказания
    if confidence_score < 0.6:
        predicted_label = "неопознан"
    else:
        predicted_label = output[np.argmax(prediction)]

    print(f"Classified {os.path.basename(cropped_img_path)} as {predicted_label} with confidence {confidence_score:.2f}")

    # Записываем результаты в текстовый файл
    with open(results_file, 'a', encoding="utf-8") as results_f:
        results_f.write(f"{os.path.basename(cropped_img_path)} {predicted_label}\n")

    # Записываем class_id или "неопознан" в class_ids_file
    with open(class_ids_file, 'a', encoding="utf-8") as f:
        if confidence_score >= 0.70:
            f.write(f"{output[np.argmax(prediction)]}\n")
        else:
            f.write("неопознан\n")


def process_images_from_folder(yolo_model, model, output, target_size, processed_files, processed_files_log,
                               output_folder, results_file, class_ids_file):
    current_time = time.time()
    for filename in os.listdir('.'):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if filename in processed_files:
                continue

            print(f"Processing file: {filename}")
            img = cv2.imread(filename)
            results = yolo_model(img)
            person_count = 0  # Счетчик людей на фотографии
            for result in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0:  # Check if the detected object is a person (class 0 in YOLO)
                    person_count += 1
                    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                    cropped_img_path = os.path.join(output_folder, f"{filename}_person_{person_count}.jpg")
                    cv2.imwrite(cropped_img_path, cropped_img)

                    # Классифицируем обрезанную фотографию
                    classify_cropped_images(model, output, target_size, cropped_img_path, results_file, class_ids_file)

            processed_files[filename] = current_time

    try:
        with open(processed_files_log, 'w', encoding="utf-8") as f_log:
            json.dump(processed_files, f_log)
    except PermissionError:
        print(f"Ошибка: Нет разрешения на запись в файл {processed_files_log}. Проверьте права доступа.")


if __name__ == "__main__":
    os.chdir("photo")
    csv_path = 'labels.csv'
    unique_labels = extract_unique_labels(csv_path)
    output = unique_labels.tolist()
    print(f"Уникальные значения: {output}")

    model = load_model('first_model.h5')
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    processed_files_log = os.path.join(os.getcwd(), "processed_files.json")

    if os.path.exists(processed_files_log):
        with open(processed_files_log, 'r', encoding="utf-8") as f:
            processed_files = json.load(f)

    output_folder = "photo_network"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_size = (250, 250)

    results_file = "classification_results.txt"
    #if os.path.exists(results_file):
        #os.remove(results_file)  # Удаляем файл, если он существует, чтобы начать с чистого листа

    class_ids_file = "class_ids.txt"
    #if os.path.exists(class_ids_file):
        #os.remove(class_ids_file)  # Удаляем файл, если он существует, чтобы начать с чистого листа

    while True:
        process_images_from_folder(yolo_model, model, output, target_size, processed_files, processed_files_log,
                                   output_folder, results_file, class_ids_file)
        os.chdir('..')

        #time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
