import cv2
import mediapipe as mp
import random

colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 0), (255, 0, 255)]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

webcam = cv2.VideoCapture(0)

while webcam.isOpened:
    ret, frame = webcam.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x,y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, random.choice(colors), 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()