import os
import numpy as np
import face_recognition
import cv2 as cv

dir = r"D:\Python workspace\Face Recognition Project\Advanced Face Recognition\pictures"
classNames = [name for name in os.listdir(dir)]

features = []

def create_train():
    for person in classNames:
        path = os.path.join(dir,person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            features.append(img_array)
            break

create_train()
print("training done---------------")

def find_encodings(features):
    encodeList = []
    for img in features:
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encode_list_known = find_encodings(features)
print(len(encode_list_known))
print("Encoding complete-------------")

video = cv.VideoCapture(0)

while True:
    isTrue,frame = video.read()
    name = "UNKNOWN"
    imgS = cv.resize(frame,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS,cv.COLOR_BGR2RGB)

    faceLocs = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS,faceLocs)

    for encode, faceLoc in zip(encodes,faceLocs):
        matches = face_recognition.compare_faces(encode_list_known,encode)
        faceDis = face_recognition.face_distance(encode_list_known,encode)

        matchIndex = np.argmin(faceDis)

        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv.putText(frame,name,(x1,y1-20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
        
        else:
            cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv.putText(frame,name,(x1,y1-20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    
    cv.imshow("livecam",frame)
    cv.waitKey(1)