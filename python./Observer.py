import numpy as np
import cv2
import os
import time

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Установка HOG детектора и экстрактора дескрипторов HOG
hog_detector = cv2.HOGDescriptor()
hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Функция для вычисления HOG-дескрипторов для вырезанного региона
def get_hog_descriptor(image):
    resized_image = cv2.resize(image, (64, 128))  # Размеры должны совпадать с требованиями HOG
    hog_descriptor = hog_detector.compute(resized_image)
    return hog_descriptor.flatten()

# Настройки видеозахвата
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

# Создание папок для хранения фотографий и скриншотов, если они не существуют
photo_dir = 'photo'
screenshot_dir = 'screenshot'

if not os.path.exists(photo_dir):
    os.makedirs(photo_dir)

# Проверка на существование папки screenshot и создание её, если её нет
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# Создание скриншота при запуске программы
ret, frame = cap.read()
if ret:
    timestamp = time.strftime("%Y%m%d%H%M%S")
    screenshot_path = os.path.join(screenshot_dir, f'screenshot_{timestamp}.jpg')
    cv2.imwrite(screenshot_path, frame)

known_persons = []
frame_interval = 30  # Интервал между снимками в кадрах
tolerance = 0.8  # Порог для косинусного сходства

previous_frame = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Пропускаем кадры вплоть до достижения нужного интервала
    if frame_count % frame_interval != 0:
        continue

    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Проверка изменения в кадре
    if previous_frame is not None:
        diff = cv2.absdiff(previous_frame, gray)
        if np.sum(diff) < 1e5:  # Например, если изменение в кадре незначительное - можно изменить значение по необходимости
            continue

    previous_frame = gray  # Обновление предыдущего кадра

    rects, weights = hog_detector.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16), scale=1.05)
    current_person_descriptors = []

    for (x, y, w, h) in rects:
        person_roi = frame_resized[y:y + h, x:x + w]
        descriptor = get_hog_descriptor(person_roi)
        current_person_descriptors.append(descriptor)

    if current_person_descriptors:
        if not known_persons:
            for descriptor in current_person_descriptors:
                known_persons.append(descriptor)
                timestamp = time.strftime("%Y%m%d%H%M%S")
                photo_path = os.path.join(photo_dir, f'photo_{timestamp}.jpg')
                cv2.imwrite(photo_path, frame_resized)
        else:
            is_any_new_person = False
            for current_descriptor in current_person_descriptors:
                if all(cosine_similarity(current_descriptor, known_descriptor) < tolerance for known_descriptor in known_persons):
                    is_any_new_person = True
            if is_any_new_person:
                known_persons = current_person_descriptors  # Обновление известных силуэтов
                timestamp = time.strftime("%Y%m%d%H%M%S")
                photo_path = os.path.join(photo_dir, f'photo_{timestamp}.jpg')
                cv2.imwrite(photo_path, frame_resized)

    # Отрисовка рамок
    for (x, y, w, h) in rects:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
