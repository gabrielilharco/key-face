import cv2

class FaceDetector():
	def __init__(self, haarCascadeFilePath, videoCapture):
		try:
			self.faceCascade = cv2.CascadeClassifier(haarCascadeFilePath)
		except:
			raise Exception("Error creating haar cascade classifier. Are you sure file " + haarCascadeFilePath + " exists?")
		self.videoCapture = videoCapture
		self.tickFrequency = cv2.getTickFrequency


	def step(self):
		ret, img = self.videoCapture.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = self.faceCascade.detectMultiScale(gray, 1.1, 3)
		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
			roi_gray = gray[y:y+h, x:x+h]
			roi_color = img[y:y+h, x:x+h]

		cv2.imshow('img', img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			return False
		return True
