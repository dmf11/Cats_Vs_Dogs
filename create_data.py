# cnn_dogsvcats.py

# https://www.kaggle.com/c/dogs-vs-cats

# In this .py file, I will create an algorithm to classify wheather images contain either a dog or cat.

# We will be taking in pre-labeled images, feeding them through a Convolutional Neural Network which will 
# classify the images with a high percentage of accuraccy.

#==============================================================================================================

# Pre-processing

	# Now that we have downloaded the data, we need to process it before we can feed it to our CNN.
		# This entails:
			# formatting all images to a standard size
			# grayscale all images which will help speed up classification
			# creating a 'one-hot array', the labels cat and dog will not be usefull here for our CNN


import numpy as np # - for dealing with np arrays
#pip install opencv-python
import cv2 # - for working with images
import os # - for accessing directories


TRAIN_DIR = '/home/david/Documents/Work/PythonStuff/ML/TensorFlow Projects/DogsVCats/Train_Processed/'
TEST_DIR = '/home/david/Documents/Work/PythonStuff/ML/TensorFlow Projects/DogsVCats/Test_Processed/'

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