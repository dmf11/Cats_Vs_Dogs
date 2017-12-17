# catsvdogs.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from create_data import load_data, load_test_data


#load_data()
(X_train, y_train, X_test, y_test) = load_data()

# Plotting a sample image with label
#plt.imshow(X_train[0])
#plt.title(y_train[0])
#plt.show()


# Transforming dataset from (n, width, height) to (n, depth, width, height)
X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)

# The second part of the processing is to convert data type to float32 and normalise data values
# to the range of (0 - 1) instead of being (0 - 255)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


def catdogCNN():

	model = Sequential()
	 
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 1)))
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))	 
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	 
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return(model)

def run_catdogCNN():

	model = catdogCNN()

	model.fit(X_train, y_train, batch_size=32, epochs=9, verbose=1, shuffle=True)
		 
	score = model.evaluate(X_test, y_test, verbose=0)
	print(score)

	# Predicting the unlabeled set of images: 
	test_ids, test_imgs = load_test_data()
	test_imgs = test_imgs.reshape(test_imgs.shape[0], 50, 50, 1)
	test_imgs /= 255	
	
	predictions = model.predict(test_imgs, verbose=1)

	img_ids = []
	labels = []

	for i in range(0,len(predictions)):
		if predictions[i, 0] >= 0.5:
			print('Dog :', predictions[i, 0])
			# id + label
			img_ids.append(test_ids[i])
			labels.append(1)
		else:
			print('Cat : ', predictions[i,0])
			img_ids.append(test_ids[i])
			labels.append(0)

	output_df = pd.DataFrame()
	output_df['ID'] = img_ids
	output_df['Label'] = labels
	output_df.to_csv('submission.csv')

run_catdogCNN()