from faces.face_detector import FaceDetector
import cv2 

cap = cv2.VideoCapture(0)

fd = FaceDetector("model/faces/haarcascade_frontalface.xml", cap)

while fd.detectFace():
	continue

cap.release()
cv2.destroyAllWindows()

