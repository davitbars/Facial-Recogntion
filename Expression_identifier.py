import pytesseract
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

pytesseract.pytesseract.tesseract_cmd = r'c:\users\davit\appdata\local\programs\tesseract-ocr\tesseract.exe'

img = cv2.imread('plate.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.resize(gray, (0,0), fx=0.3,fy=0.3)

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 20, 200) #Edge detection

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 100, True)
    if len(approx) == 4:
        location = approx
        break

print(location)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask) 

#(x,y) = np.where(mask==255)
#(x1, y1) = (np.min(x), np.min(y))
#(x2, y2) = (np.max(x), np.max(y))
#cropped_image = gray[x1:x2+1, y1:y2+1]

cv2.imshow('img',new_image)
cv2.waitKey()

#text = pytesseract.image_to_string(img)


