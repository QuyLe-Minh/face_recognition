import face_recognition
import os
import cv2 as cv
from face_recognition.api import face_locations
import numpy as np
import dlib

img = face_recognition.load_image_file(r"Advanced Face Recognition\pictures\Minh Quy\MinhQuy (3).JPG")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file(r"Advanced Face Recognition\pictures\Minh Quy\current_Quy.jpg")
img_test = cv.cvtColor(img_test,cv.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img)[0]
encode = face_recognition.face_encodings(img)[0]
cv.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLoc_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv.rectangle(img_test,(faceLoc_test[3],faceLoc_test[0]),(faceLoc_test[1],faceLoc_test[2]),(255,0,0),2)

result = face_recognition.compare_faces([encode],encode_test)
dis = face_recognition.face_distance([encode],encode_test)
print(result,dis)

cv.imshow("Quy",img)
cv.imshow("Quy test",img_test)

cv.waitKey(0)