from deepface import DeepFace
import cv2 as cv

cap = cv.VideoCapture("E:\Programs\Festika2025\meet2\Vid\Jiwoo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = DeepFace.extract_faces(frame, detector_backend='yolov8', enforce_detection=False)

    for face in faces:
        area_wajah = face['facial_area']
        x,y,w,h = area_wajah['x'], area_wajah['y'], area_wajah['w'], area_wajah['h']
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("Wajah Terdeteksi",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()