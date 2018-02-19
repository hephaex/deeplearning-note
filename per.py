import keras
from keras.models import load_model
import sys
import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
	imgcpy = np.copy(img)
	imgcpy *= 128
	imgcpy += 128
	imgcpy = imgcpy.astype('uint8')
	imgcpy = np.swapaxes(imgcpy,0,2)
	imgcpy = np.swapaxes(imgcpy,0,1)
	plt.imshow(imgcpy)
	plt.show()


def cyclic(value, lb, ub):
	if value < lb:
		return value + (ub-lb)
	elif value > ub:
		return value - (ub-lb)
	else:
		return value


def top_k_predicitons(model, X, k):
	proba = model.predict_proba(X, verbose=0)
	labels = np.argsort(proba, axis=1)
	selected_labels = (labels[:,:k])[0]
	return selected_labels


def perturbed(img, x, y, p):
	perturbed = np.copy(img)
	perturbed[:,x,y] = np.sign(perturbed[:,x,y]) * p
	return perturbed


def randadv(model, img, label, p, U):
	# Expects image in th-ordering
	# Assumes img is a good image (as defined in the paper)
	critical = 0
	for _ in range(U):
		x = np.random.choice(img.shape[1])
		y = np.random.choice(img.shape[2])
		perturb = perturbed(img, x, y, p)
		if label not in top_k_predicitons(model, perturb.reshape(1, 3, 32, 32), 1):
			critical += 1
	return float(critical) / U


def locsearchadv(model, img, p, r, d, t, k, R, label, show=False):
	dim1, dim2 = img.shape[1], img.shape[2]
	num_pixels = int(dim1*dim2*0.1)
	PX, PY = np.random.choice(range(int(dim1)),num_pixels), np.random.choice(range(int(dim2)),num_pixels)
	i = 1
	while i <= R:
		I = np.copy(img)
		# Computing the function g using the neighborhood
		L = []
		for j in range(len(PX)):
				L.append(perturbed(I, PX[j], PY[j], p))
		L = np.array(L)
		scores = model.predict_proba(L)[:,label]
		sorted_L = np.argsort(scores)
		PX = (PX[sorted_L])[:t]
		PY = (PY[sorted_L])[:t]
		# Generation of the perturbed image I
		for j in range(len(PX)):
			I = perturbed(I, PX[j], PY[j], r)
		# Check whether the perturbed image I is an adversarial image
		predictions = top_k_predicitons(model, I.reshape(1, 3, 32, 32), k)
		if label not in predictions:
			return (i, True, label, I)
		# Update a neighborhood of pixel locations for the next round
		PX_ , PY_ = [], []
		for j in range(len(PX)):
			for k in range(-d,d+1):
				x_co = PX[j] + k
				if x_co >0 and x_co < I.shape[1]:
					for l in range(-d,d+1):
						y_co = PY[j] + l
						if y_co > 0 and y_co < I.shape[2]:
							PX_.append(x_co)
							PY_.append(y_co)
		if show:
			show_image(I)
		PX, PY = np.array(PX_), np.array(PY_)
		i += 1
	return (i, False, -1, None)


def perturb_images(model, images, labels, p, r, d, t, k, R):
	n_images = len(labels)
	success_count = 0.0
	total_count = 0
	perturbed_images = []
	perturbed_labels = []
	valid_labels = []
	for i in range(n_images):
		image = images[i]
		label = np.argmax(labels[i])
		count, success, new_label, noisy_image = locsearchadv(model, image, p, r, d, t, k, R, label)
		total_count += count
		if success:
			perturbed_images.append(noisy_image)
			perturbed_labels.append(new_label)
			success_count += 1.0
			valid_labels.append(i)
	print("\n%f percent images were successfully perturbed"%(100*success_count/n_images))
	return np.array(perturbed_images), np.array(perturbed_labels), valid_labels, total_count


if __name__ == "__main__":
	try:
		model = load_model(sys.argv[1])
		image = np.load(sys.argv[2])
	except:
		print "python " + sys.argv[1] + " <model.h5> <image.npy>"
		exit(-1)
	try:
		label = int(sys.argv[2].split('-')[1].split('.')[0])
	except:
		print "Imagename not in specified (auto-generated) format"
		exit(-1)
	if image.shape[1] != 3:
		image = np.transpose(image, (2,0,1))
	print(locsearchadv(model, image, p=1, r=1, d=3, t=10, k=2, R=4, label=label))
