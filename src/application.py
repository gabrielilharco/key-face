from landmarks.load_data import load


X, y = load("data/landmarks/training.csv")
print "Shape of X:", X.shape
print "Shape of y:", y.shape