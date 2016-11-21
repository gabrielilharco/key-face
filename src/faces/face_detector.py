import cv2
import settings

class FaceDetector():
	def __init__(self, haarCascadeFilePath, videoCapture):
		try:
			self.faceCascade = cv2.CascadeClassifier(haarCascadeFilePath)
			print self.faceCascade
		except:
			raise Exception("Error creating haar cascade classifier. Are you sure file " + haarCascadeFilePath + " exists?")
		self.videoCapture = videoCapture
		self.foundFace = False
		self.trackedFace = None
		self.trackedFaceROI = None
		self.templateMatchingStartTime = cv2.getTickCount()
		self.templateMatchingCurrentTime = cv2.getTickCount()
		self.isTemplateMatchingRunning = False
		self.img = None

	def limit(self, val, inf, sup):
		return max(inf,min(val,sup))

	def limit_rect(self, rect, img):
		(x,y,w,h) = rect
		(img_height, img_width) = img.shape
		limited_x = self.limit(x, 0, img_height)
		limited_y = self.limit(y, 0, img_width)
		limited_width = self.limit(w, 0, img_width-limited_x)
		limited_height = self.limit(h, 0, img_height-limited_y)
		return (limited_x, limited_y, limited_width, limited_height)

	def area(self, face):
		return (face[2])*(face[3])

	def largestFace(self, faces):
		if len(faces) == 0:
			return None
		largest = faces[0]
		for face in faces[1:]:
			if  self.area(face) > self.area(largest):
				largest = face
		return largest

	def getSubRect(self, img, rect):
		(x, y, w, h) = rect
		return img[y:y+h, x:x+w]

	def scaleRect(self, rect, img, factor):
		(x, y, w, h) = rect
		roi = (int(x+w*(1-factor)/2), int(y+h*(1-factor)/2), int(w*factor), int (h*factor))
		return self.limit_rect(roi, img)

	def detectCascade(self, img, roiOnly=False):
		if roiOnly:
			searchArea = self.getSubRect(img, self.trackedFaceROI)
		else:
			searchArea = img

		width = searchArea.shape[0]
		faces = self.faceCascade.detectMultiScale(searchArea,
			scaleFactor = settings.cascadeScaleFactor,
			minNeighbors = settings.cascadeMinNeighbors,
			minSize = (int(width*settings.minimumFaceSize), int(width*settings.minimumFaceSize)),
			maxSize = (int(width*settings.maximumFaceSize), int(width*settings.maximumFaceSize)))

		if len(faces) == 0:
			if roiOnly and not self.isTemplateMatchingRunning:
				self.isTemplateMatchingRunning = True
				self.templateMatchingStartTime = cv2.getTickCount()
			elif not roiOnly:
				self.foundFace = False
				self.trackedFace = None;
			return

		self.foundFace=True
		# track only the largest face
		self.trackedFace = self.largestFace(faces)
		# adjust face position if necessary
		if roiOnly:
			self.trackedFace[0] += self.trackedFaceROI[0]
			self.trackedFace[1] += self.trackedFaceROI[1]
		self.trackedFaceTemplate = self.scaleRect(self.trackedFace, img, 0.5)
		self.trackedFaceROI = self.scaleRect(self.trackedFace, img, 1.5)

	def detectTemplateMatching(self, img):
		self.templateMatchingCurrentTime = cv2.getTickCount()
		duration = (self.templateMatchingCurrentTime - self.templateMatchingStartTime)/cv2.getTickFrequency()
		if duration > settings.templateMatchingDuration or self.trackedFaceTemplate[2] == 0 or self.trackedFaceTemplate[3] == 0:
			self.foundFace = False
			self.isTemplateMatchingRunning = False
			return

		faceTemplate = self.getSubRect(img, self.trackedFaceTemplate)
		roi = self.getSubRect(img, self.trackedFaceROI)
		match = cv2.matchTemplate(roi, faceTemplate, cv2.TM_SQDIFF_NORMED)
		cv2.normalize(match, match, 0, 1, cv2.NORM_MINMAX, -1)


		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(match)
		foundTemplate = (
			minLoc[0] + self.trackedFaceROI[0],
			minLoc[1] + self.trackedFaceROI[1],
			self.trackedFaceTemplate[2],
			self.trackedFaceTemplate[3])

		self.trackedFaceTemplate = foundTemplate
		self.trackedFace = self.scaleRect(self.trackedFaceTemplate, img, 2)
		self.trackedFaceROI = self.scaleRect(self.trackedFace, img, 1.5)

	def resize(self, img):
		original_height = img.shape[0]
		original_width = img.shape[1]
		self.scale = float(min(settings.resizedWidth, original_width)) / original_width
		return cv2.resize(img, (int(self.scale*original_width), int(self.scale*original_height)))

	def detectFace(self):
		ret, img = self.videoCapture.read()
		# reescaling if necessary, mantaining aspect ratio
		if img.shape[1] != settings.resizedWidth:
			img = self.resize(img)
		# getting only gray scale components
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if not self.foundFace:
			self.detectCascade(gray_img, roiOnly=False) # detect on whole image (using haar cascades)
		else:
			self.detectCascade(gray_img, roiOnly=True) # detect on ROI (using haar cascades)
			if self.isTemplateMatchingRunning:
				self.detectTemplateMatching(gray_img) # detect using template matching
		if self.foundFace:
			cv2.rectangle(img,
				(self.trackedFace[0], self.trackedFace[1]),
				(self.trackedFace[0]+self.trackedFace[2], self.trackedFace[1]+self.trackedFace[3]),
				(0,255,0), 3)
		#Local copy
		self.img = img
		self.gray_img = gray_img

		cv2.imshow('img', img)

	def getFaceImage(self):
		return self.getSubRect(self.img, self.trackedFace)

	def getGrayFaceImage(self):
		return self.getSubRect(self.gray_img, self.trackedFace)




