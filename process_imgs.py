# process_imgs.py

# This .py file will:
	# - resizing images
	# - grayscale images
	# - save the processed images into a new directory


import cv2 # - for working with images
import os # - for accessing directories
from tqdm import tqdm # - progress bar for tasks added onto loops

IMAGE_H = 50
IMAGE_W = 50
TRAIN_DIR = '/TRAIN/'
TEST_DIR = '/TEST/'

def process_train_data():
	if(os.path.isdir(TRAIN_PROCESSED) == True):
		print('path already exists')
	else:
		os.mkdir(TRAIN_PROCESSED)
		for i in tqdm(os.listdir(TRAIN_DIR)):	
			img = cv2.imread(TRAIN_DIR + i, 0)		
			img = cv2.resize(img, (IMAGE_H, IMAGE_W))
			#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) - No longer needed, providing the parameter '0' when reading the  image in will grayscale the image.
			new_path = TRAIN_PROCESSED + i # Saving the processed image into the directory 'new_path'
			cv2.imwrite(new_path, img)
process_train_data()

def process_test_data():
	if(os.path.isdir(TEST_PROCESSED) == True):
		print('TEST path already exists')
	else:
		os.mkdir(TEST_PROCESSED)
		for i in tqdm(os.listdir(TEST_DIR)):			
			img = cv2.imread(TEST_DIR + i, 0)
			img = cv2.resize(img, (IMAGE_H, IMAGE_W))
			cv2.imwrite(TEST_PROCESSED + i, img)
process_test_data()
