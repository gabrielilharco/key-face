from faces.face_detector import FaceDetector
import cv2 

cap = cv2.VideoCapture(0)

fd = FaceDetector("model/faces/haarcascade_frontalface.xml", cap)

running = True
while running:
	fd.detectFace()
	key = cv2.waitKey(30) & 0xff
	if key == 27:
		running = False

	

cap.release()
cv2.destroyAllWindows()

