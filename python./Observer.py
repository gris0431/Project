import numpy as np
import cv2
import dlib
import os
import time

# Загрузка моделей и инициализация детектора
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH1 = "./shape_predictor_68_face_landmarks.dat"
PREDICTOR_PATH2 = "./dlib_face_recognition_resnet_model_v1.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH1)
face_rec_model = dlib.face_recognition_model_v1(PREDICTOR_PATH2)

# Запуск видеозахвата
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

# Создание каталога для фотографий, если он не существует
if not os.path.exists('photo'):
    os.makedirs('photo')

known_faces = []  # Известные лица
tolerance = 0.6  # Порог схожести лиц для определения уникальности
forget_after_seconds = 5  # Время в секундах, после которого лицо будет забыто
forget_timer = {}  # Таймер забывания лиц

# Бесконечный цикл обработки видеопотока
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    current_face_descriptors = []  # Дескрипторы текущих лиц

    # Определение дескрипторов для каждого лица в кадре
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
        current_face_descriptors.append(face_descriptor)

    # Удаление лиц, которых больше нет в кадре, из списка известных лиц
    now = time.time()  # Текущее время
    forgettable_faces = [face for face, last_seen in forget_timer.items() if now - last_seen > forget_after_seconds]
    for face in forgettable_faces:
        known_faces.remove(face)
        del forget_timer[face]

    # Определение новых лиц и съемка фотографии
    for face_descriptor in current_face_descriptors:
        is_new_face = all(
            np.linalg.norm(np.array(face_descriptor) - np.array(known_face)) > tolerance for known_face in known_faces)
        if is_new_face:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            photo_path = os.path.join('photo', f'photo_{timestamp}.jpg')
            cv2.imwrite(photo_path, frame)
            known_faces.append(face_descriptor)  # Добавление нового лица в список известных лиц
            forget_timer[face_descriptor] = time.time()  # Установка времени забывания для нового лица

    # Отображение кадра
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
