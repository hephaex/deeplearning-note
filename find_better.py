from keras.datasets import cifar100
import numpy as np
from keras.models import load_model
import sys


def image_format(X):
	X = np.swapaxes(X,0,2)
	X = np.swapaxes(X,0,1)
	return X


def get_processed_data():
	X, y = get_data()
	return process_data(X), y


def get_data():
	(X_train, y_train), (X_test, y_test) = cifar100.load_data()
	return X_test, y_test


def process_data(X):
	X = X.astype('float32')
	X -= 128
	X /= 128
	X = np.transpose(X, (0,3,1,2))
	return X


def find_good_image(model, X, y):
	rand_indices = np.random.shuffle(np.arange(len(y)))
	X_shuff, y_shuff = X[rand_indices], y[rand_indices]
	y_ = model.predict(X)
	skip=0
	for i in range(len(y)):
		if y[i] == np.argmax(y_[i]):
			if skip==0:
				skip += 1
				continue
			print "Found a good image!"
			return image_format(X[i]), np.argmax(y_[i])


if __name__ == "__main__":
	model = load_model(sys.argv[1])
	X, y = get_processed_data()
	image, label = find_good_image(model, X, y)
	np.save("good_image-" + str(label), image)
