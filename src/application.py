from faces.face_detector import FaceDetector
# from landmarks.test_net import *
import cv2
import numpy as np
import sys
sys.setrecursionlimit(10000)

##########################
##########################
##########################


from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb


##########################
##########################
##########################


import theano

def float32(k):
    return np.cast['float32'](k)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


#####################
#####################
#####################


#X, y = load2d()  # load 2-d data
#net5.fit(X, y)

import cPickle as pickle

model = pickle.load(open("landmarks/models/net5.pickle", 'rb'))

#model = pickle.load(open(PICKLE_MODELS_PATH + "/net5.pickle"))

class LandmarkPredictor(object):

    def __init__(self):
        self.model = model

    def predict(self, img):
        resized_img = cv2.resize(img, (96, 96))
        X = resized_img.reshape(1, 1, 96, 96)
        X = X / 255.  # scale pixel values to [0, 1]
        X = X.astype(np.float32)
        y = self.model.predict(X)
        print y

cap = cv2.VideoCapture(0)

fd = FaceDetector("../model/faces/haarcascade_frontalface.xml", cap)
lp = LandmarkPredictor()

running = True
while running:
	fd.detectFace()

	if fd.foundFace:
		faceImage = fd.getGrayFaceImage()
		lp.predict(faceImage)

	key = cv2.waitKey(30) & 0xff
	if key == 27:
		running = False



cap.release()
cv2.destroyAllWindows()

