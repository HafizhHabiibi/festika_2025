from deepface import DeepFace
import cv2 as cv

img = cv.imread('E:\Programs\Festika2025\meet2\Images\carmen.jpg')
face = DeepFace.extract_faces(img, detector_backend='yolov8')

print(f"Jumlah wajah terdeteksi : {len(face)}" )

for i, f in enumerate (face):
    area_wajah = f['facial_area']

    x,y,w,h = area_wajah['x'], area_wajah['y'], area_wajah['w'], area_wajah['h']

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow("Deteksi", img)
cv.waitKey(0)
cv.destroyAllWindows()