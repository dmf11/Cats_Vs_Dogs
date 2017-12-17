# Pre-processing

import numpy as np
import cv2
import os

TRAIN_DIR = '/Train_Processed/'
TEST_DIR = '/Test_Processed/'

def create_labels(imageName):

	label = imageName.split('.')[-3]

	if label == 'cat':
		return(0)
	elif label == 'dog':
		return(1)

def load_data():

	images = []
	labels = []

	for file in os.listdir(TRAIN_DIR):		
		img = cv2.imread(TRAIN_DIR + file, 0)
		labels.append(create_labels(file))
		images.append(img)	
	X_train = images[:-5000]
	y_train = labels[:-5000]
	X_test = images[-5000:]
	y_test = labels[-5000:]

	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	X_test = np.asarray(X_test)
	y_test = np.asarray(y_test)

	return(X_train, y_train, X_test, y_test)


def load_test_data():

	test_images = []
	test_ids = []
	for file in os.listdir(TEST_DIR):

		img = cv2.imread(TEST_DIR + file, 0)
		test_images.append(img)
		test_ids.append(get_test_ids(file))
			
	test_images = np.asarray(test_images)
	return(test_ids, test_images)


def get_test_ids(fileName):
	return(fileName.split('.')[0])
