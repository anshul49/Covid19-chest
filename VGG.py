
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications import vgg16
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

model = vgg16.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    classifier_activation="softmax",
)
for layers in model.layers:
    layers.trainable=False
    
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(256, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(class1)

model = Model(inputs = model.inputs, outputs = output)


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

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

#model Fitting

hist = model.fit(
	train_generator,
	steps_per_epoch=10,
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


##PLOTTING LOSS AND ACCURACY


plt.figure(1, figsize = (16,10)) 
    
plt.subplot(221)  
plt.plot(hist.history['accuracy'])  
plt.plot(hist.history['val_accuracy'])  
plt.title('VGG16 accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test']) 
    
plt.subplot(222)  
plt.plot(hist.history['loss'])  
plt.plot(hist.history['val_loss'])  
plt.title('VGG16 loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test']) 

plt.show()

y_actual=[]
y_test=[]
y_actual = np.array(y_actual)
y_test = np.array(y_test)



for i in os.listdir('/Users/xd_anshul/Desktop/Research/Major/CovidDataset/Test/Normal/'):
	img=image.load_img('/Users/xd_anshul/Desktop/Research/Major/CovidDataset/Test/Normal/'+i,target_size=(224,224))
	img=image.img_to_array(img)
	img=np.expand_dims(img,axis=0)
	p=model.predict(img)
	y_test.concatenate(p[0,0])
	y_actual.concatenate(1)

for i in os.listdir('/Users/xd_anshul/Desktop/Research/Major/CovidDataset/Test/Covid/'):
	img=image.load_img('/Users/xd_anshul/Desktop/Research/Major/CovidDataset/Test/Covid/'+i,target_size=(224,224))
	img=image.img_to_array(img)
	img=np.expand_dims(img,axis=0)
	p=model.predict(img)
	y_test.concatenate(p[0,0])
	y_actual.concatenate(0)

y_actual=np.array(y_actual)
y_test=np.array(y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_actual,y_test)

import seaborn as sns

sns.heatmap(cm,cmap='plasma',annot=True)

Y_pred = model.predict(validation_generator, steps=2+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())   
report = classification_report(target_names, y_pred, target_names=class_labels)
print(report) 



