import cv2 as cv
import numpy as np
import os

harr_cascade = cv.CascadeClassifier(r'E:\Programs\Festika2025\meet3\haarcascade.xml')

features = []
labels = []
peoples = ["Cut Syifa", "Gofar Hilman", "Raditya Dika", 
            "Raffi Ahmad", "Jiwoo"]

def create_train():
    path = r'E:\Programs\Festika2025\meet3\train'

    for people in peoples:
        path_people = os.path.join(path, people)
        label = peoples.index(people)

        for img in os.listdir(path_people):
            img_path = os.path.join(path_people, img)
            img_read = cv.imread(img_path)
            img_gray = cv.cvtColor(img_read, cv.COLOR_BGR2GRAY)

            faces = harr_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                # cv.rectangle(img_read, (x, y), (x + w, y + h), thickness=3, color=(0, 255, 0))
                # cv.imshow("test", img_read)
                # break
                face = img_gray[y:y+h, x:x+w]
                features.append(face)
                labels.append(label)

            # break

create_train()

features = np.array(features, dtype=object)
labels = np.array(labels)

face_recognition = cv.face.LBPHFaceRecognizer_create()
face_recognition.train(features, labels)

face_recognition.save('face_recog.yml')
print('train model selesai')