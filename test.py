import numpy as np
import cv2 as cv
import os

dir = r"D:\Face Recognition Project\Basic Face Recognition\pictures"
people = [name for name in os.listdir(dir)]

haar_cascade = cv.CascadeClassifier("D:\Face Recognition Project\Basic Face Recognition\haar_face.xml")
features = np.load("features.npy",allow_pickle=True)
labels = np.load("labels.npy",allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")
