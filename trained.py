import os
import numpy as np
import face_recognition
import cv2 as cv

dir = r"Advanced Face Recognition\pictures"
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

features = np.array(features,dtype="object")
encode_list_known = np.array(encode_list_known,dtype="object")


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,encode_list_known)

face_recognizer.save("face_trained.yml")
np.save("features.npy",features)
np.save("labels.npy",encode_list_known)