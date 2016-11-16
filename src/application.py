from faces.face_detector import FaceDetector
from landmarks.landmark_predictor import LandmarkPredictor
import cv2

cap = cv2.VideoCapture(0)

fd = FaceDetector("model/faces/haarcascade_frontalface.xml", cap)
lp = LandmarkPredictor()

running = True
while running:
	fd.detectFace()
	faceImage = fd.getFaceImage()
	lp.predict(faceImage)

	key = cv2.waitKey(30) & 0xff
	if key == 27:
		running = False



cap.release()
cv2.destroyAllWindows()

