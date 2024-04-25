import cv2
from mtcnn import MTCNN
import os
import time

# Получаем текущее время
current_time = time.time()

# Устанавливаем текущую рабочую директорию в папку с фотографиями (если она еще не установлена)
os.chdir('photo')

# Переменная для новой папки с размеченными и обрезанными фотографиями
output_folder = "photo_network"
# Создаем папку, если она еще не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Перебираем все файлы в текущей директории
for filename in os.listdir():
    if filename.endswith(".jpg"):
        # Получаем время последнего изменения файла (время создания фотографии)
        file_creation_time = os.path.getmtime(filename)

        # Проверяем, прошло ли менее 15 секунд с момента создания файла
        if current_time - file_creation_time < 15:
            detector = MTCNN()
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)

            # Проверяем, найдены ли лица на фото
            if len(result) > 0:
                for i, face_data in enumerate(result):
                    bounding_box = face_data['box']
                    keypoints = face_data['keypoints']

                    # Размеры рамки с лицом
                    x, y, w, h = bounding_box

                    # Увеличиваем рамку для улучшенной обработки лица
                    x -= 20
                    y -= 30
                    w += 40
                    h += 50

                    # Вырезаем область с лицом из исходного изображения
                    face_image = image[y:y + h, x:x + w]

                    # Сохраняем обрезанное изображение лица в новой папке с индексом лица в имени файла
                    cv2.imwrite(os.path.join(output_folder, f"face_{i + 1}_{filename}"),
                                cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                    print(f"Saved cropped and annotated face {i + 1} from {filename}")

            else:
                print(f"No faces found in {filename}")

        else:
            print(f"Skipping photo {filename} as it was taken more than 15 seconds ago")

os.chdir('..')