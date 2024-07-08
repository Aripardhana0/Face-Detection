#Nama : Tsalasanita Nuraeni
#NIM : 21518241013

import cv2
import time


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)


if not cam.isOpened():
    print("Error: Kamera tidak ditemukan atau gagal dibuka.")
else:
    while cam.isOpened():
        ret, img = cam.read()
        if not ret:
            break

        _time_mulai = time.time()   
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        print('waktu', time.time() - _time_mulai)
        img = cv2.resize(img, (960, 960))
        cv2.imshow("image", img)
        _key = cv2.waitKey(1)
        if _key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


