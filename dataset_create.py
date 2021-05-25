import pandas as pd

#creating a data for positive samples

file_path = "chestxray/metadata.csv"
image_path = "chestxray/images"

df = pd.read_csv(file_path)
print(df.shape)

df.head

target_dir = "Dataset/Covid" #new folder for Dataset

import os
import shutil

if not os.path.exists(target_dir):
	os.mkdir(target_dir)
	print("Covid Folder Created!")

cnt = 0

for (i,row) in df.iterrows(): #(using only xray with front view)
	if row['finding']=='COVID-19' and row['view']=="PA":
		filename = row['filename']
		image_path = os.path.join(image_path, filename)
		image_copy_path = os.path.join(target_dir, filename)
		shutil.copy2(image_path,image_copy_path)
		print('Moving image',cnt)
		cnt+=1

print(cnt)

# sampling normal images from kaggle dataset into normal folder
import random
kaggle_file_path = 'chest_xray_kaggle/train/NORMAL'
target_normal_dir = 'Dataset/Normal'

image_names = os.listdir(kaggle_file_path)

random.shuffle(image_names)

for i in range(142):
	image_name = image_names[i]
	image_path = os.path.join(kaggle_file_path,image_name)
	target_path = os.path.join(target_normal_dir, image_name)
	shutil.copy2(image_path,target_path)
