import cv2 as cv

# Load Haar Cascade dan model
harr_cascade = cv.CascadeClassifier(r'E:\Programs\Festika2025\meet3\haarcascade.xml')
face_recognition = cv.face.LBPHFaceRecognizer_create()
face_recognition.read('face_recog.yml')

# Daftar label
peoples = ["Cut Syifa", "Gofar Hilman", "Raditya Dika", "Raffi Ahmad", "Jiwoo"]

# Baca dan ubah gambar
img_path = r'E:\Programs\Festika2025\meet3\val\Jiwoo\val1.jpg'
img_read = cv.imread(img_path)
img_gray = cv.cvtColor(img_read, cv.COLOR_BGR2GRAY)

# Deteksi wajah
faces = harr_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)

# Loop setiap wajah yang terdeteksi
for (x, y, w, h) in faces:
    face = img_gray[y:y+h, x:x+w]
    label, confidence = face_recognition.predict(face)
    
    # Gambar kotak di sekitar wajah
    cv.rectangle(img_read, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # Tampilkan label dan confidence di atas kotak
    text = f"{peoples[label]}"
    cv.putText(img_read, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f'label: {peoples[label]}, confidence: {confidence:.2f}')

# Tampilkan hasil gambar dengan kotak
cv.imshow("Hasil Analisis Wajah", img_read)
cv.waitKey(0)
cv.destroyAllWindows()