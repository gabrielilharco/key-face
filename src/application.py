from faces.face_detector import FaceDetector
import cv2 

cap = cv2.VideoCapture(0)

fd = FaceDetector("data/faces/haarcascade_frontalface.xml", cap)

while fd.step():
	continue

cap.release()
cv2.destroyAllWindows()

