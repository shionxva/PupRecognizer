#Operation Urd - Stage 1
#Process images and labels from dataset
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm

#Desired output size for images, DO NOT CHANGE IN CASE OF USING MOBILENETV2
OUTPUT_SIZE : tuple[int, int] = (224, 224)

#CHANGE THESE PATHS TO YOUR OWN
#SAVE_IMAGE_PATH : str = r"D:\temp\saved images"
#SAVE_LABELS_PATH : str = r"D:\temp\saved labels\labels.txt"
TRAIN_PATH : str = r"D:\temp\train"
SAVE_IMAGE_PATH_NPY : str = r"D:\temp"

SAVE_LABELS_PATH_NPY: str = r"D:\temp"
SAVE_LEBELS_PATH_ENCODE: str = r"D:\temp\label_encoder.pkl"

#Create directories if they do not exist
os.makedirs(r"D:\temp", exist_ok=True)
#os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
#os.makedirs(SAVE_LABELS_PATH, exist_ok=True)

#load csv file
df = pd.read_csv(r"D:\temp\labels.csv")

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

allImages : list[np.ndarray] = [] #This will store a list of images as ndarrays for now
allLabels : list[str] = []

for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing images"):
    image_id : str = row.id
    breed : str = row.breed

    #Access image paths
    img_dir : str = os.path.join(TRAIN_PATH, image_id + '.jpg')

    #Read image using cv2
    img : np.ndarray = cv2.imread(img_dir, cv2.IMREAD_COLOR_RGB)
    if img is None:
        print(f"Image {img_dir} not found or could not be read.")
        continue

    rgb_img : np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resizing : np.ndarray  = cv2.resize(rgb_img, OUTPUT_SIZE, interpolation= cv2.INTER_AREA)
        
    #Store in lists
    allImages.append(resizing)
    allLabels.append(breed)

#Confirm list sizes
print(f"Image list size: {len(allImages)}")
print(f"Label list size: {len(allLabels)}")

#save into temp folder
print("Saving...")
allImages = np.array(allImages)  # Convert to 1 numpy array for contiguous memory
np.save(SAVE_IMAGE_PATH_NPY, allImages)
np.save(SAVE_LABELS_PATH_NPY, allLabels)

#Encode label
print("Encoding labels file...")
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(allLabels)  # strings → integers
with open(SAVE_LABELS_PATH_ENCODE, "wb") as f: #save labels as encoded labels file 
    pickle.dump(label_encoder, f)

print("Success")