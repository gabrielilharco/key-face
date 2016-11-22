import numpy as np
import cv2


class LandmarkPredictor(object):
    """
    Classifier that predicts 15 keypoints from an image containing
    a face.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, img):
        resized_img = cv2.resize(img, (96, 96))
        X = resized_img.reshape(1, 1, 96, 96)
        X = X / 255.  # scale pixel values to [0, 1]
        X = X.astype(np.float32)
        y = self.model.predict(X)

        return y

    def drawPoints(self, img, points, rect):
        """
        Draw the points on an image given a rectangle
        """
        points = np.transpose(points)
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            # print x, y
            size = rect[2] / 2
            center = (int(x * size + size) +
                      rect[0], int(y * size + size) + rect[1])
            cv2.circle(img, center, 3, (255, 0, 0), -1)
