train_path = "/Users/xd_anshul/Desktop/Research/Major!/CovidDataset/Train"
val_path = "/Users/xd_anshul/Desktop/Research/Major!/CovidDataset/Test"

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications import xception
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# CNN Based model in keras

model= Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(  2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])




#Data Augmentation

train_datagen = image.ImageDataGenerator(
	rescale = 1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True,
	)

test_datagen = image.ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
	'/Users/xd_anshul/Desktop/Research/Major/CovidDataset/Train',
	target_size = (224,224),
	batch_size = 10,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	'/Users/xd_anshul/Desktop/Research/Major/CovidDataset/Test',
	target_size = (224,224),
	batch_size = 10,
	class_mode='binary')

# model Fitting

hist = model.fit(
	train_generator,
	steps_per_epoch=9,
	epochs=20,
	validation_data=validation_generator,
	validation_steps=2)

# for major- 1. Class-Activation Maps & 2. Grad-CAM

#confusion_matrix!

model.save('model_adv.h5')
model.evaluate_generator(train_generator)
model.evaluate_generator(validation_generator)


model = load_model('model_adv.h5')
import os
train_generator.class_indices

y_actual=[]
y_test=[]
#y_actual = np.array(y_actual)
#y_test = np.array(y_test)



for i in os.listdir('/Users/xd_anshul/Desktop/Major/CovidDataset/Test/Normal/'):
	img=image.load_img('/Users/xd_anshul/Desktop/Major/CovidDataset/Test/Normal/'+i,target_size=(224,224))
	img=image.img_to_array(img)
	img=np.expand_dims(img,axis=0)
	p=model.predict_classes(img)
	y_test.append(p[0,0])
	y_actual.append(1)

for i in os.listdir('/Users/xd_anshul/Desktop/Major/CovidDataset/Test/Covid/'):
	img=image.load_img('/Users/xd_anshul/Desktop/Major/CovidDataset/Test/Covid/'+i,target_size=(224,224))
	img=image.img_to_array(img)
	img=np.expand_dims(img,axis=0)
	p=model.predict_classes(img)
	y_test.append(p[0,0])
	y_actual.append(0)

y_actual=np.array(y_actual)
y_test=np.array(y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_actual,y_test)

import seaborn as sns

sns.heatmap(cm,cmap='plasma',annot=True)




