import cv2

class FaceDetector():
	def __init__(self, haarCascadeFilePath, videoCapture):
		try:
			self.faceCascade = cv2.CascadeClassifier(haarCascadeFilePath)
		except:
			raise Exception("Error creating haar cascade classifier. Are you sure file " + haarCascadeFilePath + " exists?")
		self.videoCapture = videoCapture
		self.tickFrequency = cv2.getTickFrequency
		self.scale = 1
		self.foundFace = False
		self.resizedWidth = 320
		self.trackedFace = None
		self.trackedFaceROI = None
		self.templateMatchingDuration = 2
		self.templateMatchingStartTime = cv2.getTickCount()
		self.templateMatchingCurrentTime = cv2.getTickCount()
		self.isTemplateMatchingRunning = False
		self.cascadeScaleFactor = 1.1
		self.cascadeMinNeighbors = 4
		self.maximumFaceSize = 0.75
		self.minimumFaceSize = 0.25
		
	def step(self):
		ret, img = self.videoCapture.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = self.faceCascade.detectMultiScale(gray, 1.1, 15)
		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
			roi_gray = gray[y:y+h, x:x+h]
			roi_color = img[y:y+h, x:x+h]

		cv2.imshow('img', img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			return False
		return True

	def area(self, face):
		return (face[2]-face[0])*(face[3]-face[1])

	def largestFace(self, faces):
		if len(faces) == 0:
			return None
		largest = faces[0]
		for face in faces[1:]:
			if  self.area(face) > self.area(largest):
				largest = face
		print largest
		return largest

	def getFaceTemplate(self, face):
		face[0] += 0.25 * face[2] 
		face[1] += 0.25 * face[3]
		face[2] *= 0.5
		face[3] *= 0.5
		return face

	def getROI(self, face, frame):
		frameShape = frame.shape
		width = 2 * face[2]
		height = 2 * face[3]
		
		x = face[0] - 0.5 * face[2]
		y = face[1] - 0.5 * face[3]

		if x < 0:
			width += x
			x = 0
		if y < 0:
			height += y
			y = 0
		if x + width > frameShape[0]:
			width = frameShape[0] - x
		if y + height > frameShape[1]:
			height = frameShape[1] - y

		return frame[y:y+height, x:x+width]

	def detectCascade(self, img, roi_only=False):
		width = img.shape[0]
		faces = self.faceCascade.detectMultiScale(img, 
			scaleFactor = self.cascadeScaleFactor,
			minNeighbors = self.cascadeMinNeighbors,
			minSize = (int(width*self.minimumFaceSize), int(width*self.minimumFaceSize)), 
			maxSize = (int(width*self.maximumFaceSize), int(width*self.maximumFaceSize))) 

		if len(faces) == 0: 
			self.foundFace = False
			return

		self.foundFace=True
		# track only the largest face 
		self.trackedFace = self.largestFace(faces)
		self.trackedFaceTemplate = self.getFaceTemplate(self.trackedFace)
		self.trackedFaceROI = self.getROI(self.trackedFace, img)

	def detectTemplateMatching(self, frame):
		self.templateMatchingCurrentTime = cv2.getTickCount()
		duration = self.templateMatchingCurrentTime - self.templateMatchingStartTime
		if duration > self.templateMatchingDuration or frame.rows == 0 or frame.cols == 0:
			self.foundFace = False
			self.isTemplateRunning = False
			self.templateMatchingStartTime = cv2.getTickCount()
			return

		width, height = self.trackedFaceTemplate.shape[::-1]
		match = cv2.matchTemplate(self.trackedFaceROI, self.trackedFaceTemplate, cv2.TM_SQDIFF)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
		top_left = min_loc
		bottom_right = (top_left[0] + width, top_left[1] + height)
		self.trackedFace = cv2.rectangle(top_left, bottom_right, 255, 2)
		self.trackedFaceTemplate = self.getFaceTemplate(self.trackedFace)
		self.trackedFaceROI = self.getROI(self.trackedFace, frame)


	def resize(self, img):
		original_height = img.shape[0]
		original_width = img.shape[1]
		self.scale = float(min(self.resizedWidth, original_width)) / original_width
		return cv2.resize(img, (int(self.scale*original_width), int(self.scale*original_height)))

	def detectFace(self):
		ret, img = self.videoCapture.read()
		# reescaling if necessary, mantaining aspect ratio
		if img.shape[1] != self.resizedWidth:
			img = self.resize(img)

		if not self.foundFace:
			self.detectCascade(img, roi_only=False) # detect on whole image (using haar cascades)
		else:
			self.detectCascade(img, roi_only=True) # detect on ROI (using haar cascades)
			if self.isTemplateMatchingRunning:
				self.detectTemplateMatching(img) # detect using template matching
		if self.trackedFace is not None:
			cv2.rectangle(img, 
				(self.trackedFace[0], self.trackedFace[1]), 
				(self.trackedFace[0]+self.trackedFace[2], self.trackedFace[1]+self.trackedFace[3]), 
				(0,0,255), 2)
		cv2.imshow('img', img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			return False
		return True



