import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def load(filename, columns=None, random_state=42):
	"""
	Loads datafrom a csv file. 
	If <columns> is provided, only those columns are loaded.
	"""

	print "Loading data from", filename
	if columns:
		print "Selecting columns", ', '.join(columns)

	# load data as a pandas dataframe
	data = read_csv(filename)

	# select only wanted columns
	if columns:
		data = data[columns + ['Image']]

	# drop all rows with missing values
	data = data.dropna()
	print "Total samples read:", str(data.shape[0])

	# build X 
	data['Image'] = data['Image'].apply(lambda image: np.fromstring(image, sep=' ')) # transform string image in numpy array
	X = np.vstack(data['Image'].values) / 255.0 # scale pixel values to [0,1]
	X = X.astype(np.float32)
	
	# build y 
	y = data[data.columns[:-1]].values
	y = (y - 48) / 48 # scale coordinates = [0,1]

	# shuffle data
	X, y = shuffle(X, y, random_state=random_state)

	return X, y




