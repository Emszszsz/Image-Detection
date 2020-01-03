import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
import numpy as np
from skimage.transform import resize
from keras.utils import np_utils
import csv
from numpy import genfromtxt

DATA = pd.read_csv('C:/Users/littl/OneDrive/Pulpit/inzynierka/labels/classification.csv')
FRAMES = [ ]

for im_name in DATA.Frame_id:
    im = plt.imread('C:/Users/littl/OneDrive/Pulpit/inzynierka/training/' + im_name)
    FRAMES.append(im)
FRAMES = np.array(FRAMES)

labels = DATA.Class

categ_labels = np_utils.to_categorical(labels)

images = []

for i in range(0, FRAMES.shape[0]):
    im = resize(FRAMES[i], preserve_range=True, output_shape=(224,224)).astype(int)
    images.append(im)

FRAMES = np.array(images)

from keras.applications.vgg16 import preprocess_input

FRAMES = preprocess_input(FRAMES, mode='tf')

from sklearn.model_selection import train_test_split

FRAMES_train, FRAMES_valid, labels_train, labels_valid = train_test_split(FRAMES, categ_labels, test_size=0.2, random_state=42)


from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

FRAMES_train = base_model.predict(FRAMES_train)
FRAMES_valid = base_model.predict(FRAMES_valid)
""" 44, 7, 7, 512
    11, 7, 7, 512"""
FRAMES_train = FRAMES_train.reshape(44, 7*7*512)
FRAMES_valid = FRAMES_valid.reshape(11, 7*7*512)

train = FRAMES_train/FRAMES_train.max()
FRAMES_valid = FRAMES_valid/FRAMES_train.max()

model = Sequential([
    InputLayer((7*7*512,)),
    Dense(units=1024, activation='sigmoid'),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(train, labels_train, epochs=100, validation_data=(FRAMES_valid, labels_valid))

test = pd.read_csv('C:/Users/littl/OneDrive/Pulpit/inzynierka/labels/test.csv')

test_image = []
for im_name in test.Image_ID:
    im = plt.imread('C:/Users/littl/OneDrive/Pulpit/inzynierka/test/' + im_name)
    test_image.append(im)

test_im = np.array(test_image)
test_image = []
for i in range(0, test_im.shape[0]):
    test_resized = resize(test_im[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(test_resized)

test_image = np.array(test_image)

test_image = preprocess_input(test_image, mode='tf')
test_image = base_model.predict(test_image)
test_image = test_image.reshape(186, 7*7*512)
test_image = test_image/test_image.max()
predictions = model.predict_classes(test_image)
print("Screen time of ads is", predictions[predictions==1].shape[0], " seconds")
print("Screen time of no ads is", predictions[predictions==0].shape[0], " seconds")
