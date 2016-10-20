from load_data import load


X, y = load("data/training.csv")
print "Shape of X:", X.shape
print "Shape of y:", y.shape