import numpy as np
import cv2
import dlib
import os
import time

# Load models and initialize detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH1 = "./shape_predictor_68_face_landmarks.dat"
PREDICTOR_PATH2 = "./dlib_face_recognition_resnet_model_v1.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH1)
face_rec_model = dlib.face_recognition_model_v1(PREDICTOR_PATH2)

# Start video capture
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

# Create directory for photos if it doesn't exist
if not os.path.exists('photo'):
    os.makedirs('photo')

known_faces = []
tolerance = 0.6
forget_after_seconds = 5  # Set time in seconds after which a face is forgotten
forget_timer = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    current_face_descriptors = []

    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
        current_face_descriptors.append(face_descriptor)

    # Remove faces that are no longer in the frame from the known_faces list
    now = time.time()
    forgettable_faces = [face for face, last_seen in forget_timer.items() if now - last_seen > forget_after_seconds]
    for face in forgettable_faces:
        known_faces.remove(face)
        del forget_timer[face]

    for face_descriptor in current_face_descriptors:
        is_new_face = all(
            np.linalg.norm(np.array(face_descriptor) - np.array(known_face)) > tolerance for known_face in known_faces)
        if is_new_face:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            photo_path = os.path.join('photo', f'photo_{timestamp}.jpg')
            cv2.imwrite(photo_path, frame)
            known_faces.append(face_descriptor)
            forget_timer[face_descriptor] = time.time()

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)