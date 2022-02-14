import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread('assets/allFaces.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.05, 100)

for (x, y, w, h) in faces:
	cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.imshow('img', img)

cv2.waitKey()