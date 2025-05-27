#Operation Urd - Stage 1
#Process images and labels from dataset
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

IMG_size = (331,331)
full_IMG_size = (331,331,3)
trainPath = "D:/dogbreed_identification/train"

#load csv file
df = pd.read_csv("D:/dogbreed_identification/labels.csv")

# checking the df 
# print("Head of labels df")
# print("================")
# print(df.head(10))
# print(df.describe())
# print()
# print("Groups by labels")
# labelsGroup = df.groupby("breed")["id"].count()
# print(labelsGroup.head(5))

#image display test (works good)
# imgPath = "D:/dogbreed_identification/train/00a366d4b4a9bbb6c8a63126697b7656.jpg"
# img = cv2.imread(imgPath)
# cv2.imshow('dawg',img)
# cv2.waitKey(0)

#image and lables as Numpy arrays
allImages =[]
allLabels = []

for i, (images_name, breed) in enumerate (df[['id', 'breed']].values):
    img_dir = os.path.join(trainPath, images_name + '.jpg')
    print(img_dir)
    img = cv2.imread(img_dir)
    resizing = cv2.resize(img, IMG_size, interpolation= cv2.INTER_AREA)
    allImages.append(resizing)
    allLabels.append(breed)

print(len(allImages))
print(len(allLabels))

#save into temp folder
print("Saving...")
np.save("D:/temp/allDogImages.npy", allImages)
np.save("D:/temp/allDogLabels.npy", allLabels)

#Encode label
print("Encoding labels file...")
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(allLabels)  # strings → integers
with open("D:/temp/label_encoder.pkl", "wb") as f: #save labels as encoded labels file 
    pickle.dump(label_encoder, f)

print("Success")