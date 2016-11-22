from faces.face_detector import FaceDetector
from landmarks.landmark_predictor import LandmarkPredictor
from landmarks.model_loader import *
import cv2
import cPickle as pickle


cap = cv2.VideoCapture(0)

fd = FaceDetector("../model/faces/haarcascade_frontalface.xml", cap)
model = pickle.load(open("../model/landmarks/net5.pickle"))
lp = LandmarkPredictor(model)

running = True
while running:
    img = fd.detectFace()

    # Detect landmarks if image is a face is found
    if fd.foundFace:
        faceImage = fd.getGrayFaceImage()
        points = lp.predict(faceImage)
        lp.drawPoints(img, points, fd.trackedFace)

    cv2.imshow('Detected Keypoints', img)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        running = False

cap.release()
cv2.destroyAllWindows()
