import cPickle as pickle

PICKLE_MODELS_PATH = "landmarks/models"


class LandmarkPredictor(object):
    def __init__(self):
        model = pickle.load(open(PICKLE_MODELS_PATH + "/net2.pickle"))

    def predict(self, img):
        resized_img = cv2.resize(img, Size(96, 96))

        X = resized_img.reshape(-1, 1, 96, 96)

        y = model.predict(X)[0]
