import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

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
    image = image / 255.0  # Нормализация пикселей

    return image

# 4. Классификация изображения
def classify_image(model, image):
    if image is None:
        return None

    predictions = model.predict(image)
    class_idx = np.argmax(predictions, axis=1)[0]
    return class_idx

# 5. Основная функция
if __name__ == "__main__":
    target_size = (250, 250)  # Замените на размер, который ожидает ваша модель

    cap = cv2.VideoCapture(0)  # 0 - номер камеры, если у вас несколько камер, возможно, нужно будет изменить номер

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
    else:
        try:
            while True:
                # Захват изображения
                image = capture_image(cap)

                # Предобработка изображения
                preprocessed_image = preprocess_image(image, target_size)

                # Проверка формы входных данных
                if preprocessed_image is not None:
                    print(f"Форма предобработанного изображения: {preprocessed_image.shape}")

                # Классификация изображения
                class_idx = classify_image(model, preprocessed_image)

                # Вывод результата
                if class_idx is not None:
                    print(f"Класс изображения: {class_idx}")
                else:
                    print("Ошибка при классификации изображения")

                # Задержка для предотвращения перегрузки процессора
                cv2.waitKey(1000)  # Задержка в 1000 миллисекунд (1 секунда)

        except KeyboardInterrupt:
            print("Программа остановлена пользователем")

        finally:
            cap.release()
            cv2.destroyAllWindows()