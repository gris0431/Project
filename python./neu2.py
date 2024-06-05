import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Функция для извлечения уникальных значений из CSV файла
def extract_unique_labels(csv_path):
    df = pd.read_csv(csv_path)
    unique_labels = df['label'].unique()
    return unique_labels


# Основная функция
if __name__ == "__main__":
    # Извлечение уникальных значений из CSV файла
    os.chdir("photo")
    csv_path = 'labels.csv'
    unique_labels = extract_unique_labels(csv_path)

    # Запись уникальных значений в переменную output
    output = unique_labels.tolist()
    print(f"Уникальные значения: {output}")

    # 1. Загрузка модели
    model = load_model('first_model.h5')


    # 2. Захват изображения с камеры
    def capture_image(cap):
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось захватить изображение")
            return None
        return frame


    # 3. Предобработка изображения
    def preprocess_image(image, target_size):
        if image is None:
            return None

        image = cv2.resize(image, target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        return image


    # 4. Классификация изображения
    def classify_image(model, image):
        if image is None:
            return None

        predictions = model.predict(image)
        class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][class_idx]
        return class_idx, confidence


    # 5. Основная функция
    if __name__ == "__main__":
        target_size = (250, 250)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
        else:
            try:
                with open("class_ids.txt", "w", encoding="utf-8") as f:
                    while True:
                        # Захват изображения
                        image = capture_image(cap)

                        # Предобработка изображения
                        preprocessed_image = preprocess_image(image, target_size)

                        # Проверка формы входных данных
                        if preprocessed_image is not None:
                            print(f"Форма предобработанного изображения: {preprocessed_image.shape}")

                        # Классификация изображения
                        result = classify_image(model, preprocessed_image)

                        if result is not None:
                            class_idx, confidence = result
                            if confidence >= 0.70:
                                f.write(f"{output[class_idx]}\n")
                                print(f"Класс изображения: {class_idx}, Точность: {confidence:.2f}")
                            else:
                                f.write("Неопознан\n")
                                print("Неопознан")
                        else:
                            print("Ошибка при классификации изображения")

                        # Задержка для предотвращения перегрузки процессора
                        cv2.waitKey(1000)

            except KeyboardInterrupt:
                print("Программа остановлена пользователем")

            finally:
                cap.release()
                cv2.destroyAllWindows()
