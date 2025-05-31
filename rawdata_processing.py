#Operation Urd - Stage 1
#Process images and labels from dataset
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import random

#Desired output size for images, DO NOT CHANGE IN CASE OF USING MOBILENETV2
OUTPUT_SIZE : tuple[int, int] = (224, 224)

#CHANGE THESE PATHS TO YOUR OWN
TRAIN_IMAGE_PATH : str = r"D:\kaggle_dataset_train\train"

SAVE_TRAIN_IMAGE_PATH_NPY : str = r"D:\kaggle_dataset_train\X_train.npy"
SAVE_VAL_IMAGE_PATH_NPY : str = r"D:\kaggle_dataset_train\X_val.npy"
SAVE_TRAIN_LABEL_PATH_NPY: str = r"D:\kaggle_dataset_train\y_train.npy"
SAVE_VAL_LABEL_PATH_NPY: str = r"D:\kaggle_dataset_train\y_val.npy"

SAVE_LABEL_PATH_ENCODE: str = r"D:\kaggle_dataset_train\label_encoder.pkl"

#Create directories if they do not exist
os.makedirs(r"D:\kaggle_dataset_train", exist_ok=True)

#load csv file
df = pd.read_csv(r"D:\kaggle_dataset_train\labels.csv")
allLabels : np.ndarray = df['breed'].to_numpy()  # Extract labels from the DataFrame

#Encode label
print("Encoding labels file...")
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(allLabels)  # strings → integers
with open(SAVE_LABEL_PATH_ENCODE, "wb") as f: #save labels as encoded labels file 
    pickle.dump(label_encoder, f)

# Convert csv df to tuple list
data = list(zip(df['id'], df['breed']))

# Shuffle the data 
random.shuffle(data)

splitRatio : float = 0.8  # 80% for training, 20% for validation
splitIndex : int = int(len(data) * splitRatio)

trainData = data[:splitIndex]
valData = data[splitIndex:]

trainImages : list[np.ndarray] = []
valImages : list[np.ndarray] = []
trainLabels : list[str] = []
valLabels : list[str] = []


for img_id, label in tqdm(trainData, total=len(trainData), desc="Processing images"):
    #Access image paths
    img_dir : str = os.path.join(TRAIN_IMAGE_PATH, img_id + '.jpg')

    #Read image using cv2
    img : np.ndarray = cv2.imread(img_dir, cv2.IMREAD_COLOR_RGB)
    if img is None:
        print(f"Image {img_dir} not found or could not be read.")
        continue

    rgb_img : np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resizing : np.ndarray  = cv2.resize(rgb_img, OUTPUT_SIZE, interpolation= cv2.INTER_AREA)
        
    #Store in lists
    trainImages.append(resizing)
    trainLabels.append(label)

for img_id, label in tqdm(valData, total=len(valData), desc="Processing validation images"):
    #Access image paths
    img_dir : str = os.path.join(TRAIN_IMAGE_PATH, img_id + '.jpg')

    #Read image using cv2
    img : np.ndarray = cv2.imread(img_dir, cv2.IMREAD_COLOR_RGB)
    if img is None:
        print(f"Image {img_dir} not found or could not be read.")
        continue

    rgb_img : np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resizing : np.ndarray  = cv2.resize(rgb_img, OUTPUT_SIZE, interpolation= cv2.INTER_AREA)
        
    #Store in lists
    valImages.append(resizing)
    valLabels.append(label)

#Confirm list sizes
print(f"Train image list size: {len(trainImages)}")
print(f"Validation image list size: {len(valImages)}")
print(f"Train label list size: {len(trainLabels)}")
print(f"Validation label list size: {len(valLabels)}")

#save into kaggle_dataset_train folder
print("Saving...")
trainImages = np.array(trainImages)
valImages = np.array(valImages)
trainLabels = np.array(trainLabels)
valLabels = np.array(valLabels)
np.save(SAVE_TRAIN_IMAGE_PATH_NPY, trainImages)
np.save(SAVE_VAL_IMAGE_PATH_NPY, valImages)
np.save(SAVE_TRAIN_LABEL_PATH_NPY, trainLabels)
np.save(SAVE_VAL_LABEL_PATH_NPY, valLabels)

print("Success")
