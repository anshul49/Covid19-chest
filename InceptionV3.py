
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications import InceptionV3
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layers in model.layers:
    layers.trainable=False
    
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(256, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(class1)

model = Model(inputs = model.inputs, outputs = output)


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


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




plt.figure(1, figsize = (16,10)) 
    
plt.subplot(221)  
plt.plot(hist.history['accuracy'])  
plt.plot(hist.history['val_accuracy'])  
plt.title('InceptionV3 accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test']) 
    
plt.subplot(222)  
plt.plot(hist.history['loss'])  
plt.plot(hist.history['val_loss'])  
plt.title('InceptionV3 loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test']) 

plt.show()


from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict_generator(validation_generator, steps = np.ceil(validation_generator.samples / validation_generator.batch_size), verbose=1, workers=0)
y_pred=[ np.argmax(Y_pred[i]) for i in range(validation_generator.samples)]
#y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Covid', 'Normal']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))



model = load_model('model_adv.h5')
import os
train_generator.class_indices
