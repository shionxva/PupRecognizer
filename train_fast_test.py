#Operation Verthandi - Stage 2
#Create a trained model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 # New recommended version of torchvision transforms
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import pickle # For loading and saving the label encoder
import os # For file handling and path management
from tqdm import tqdm # Progress bar for training epochs
import sys # For early error handling
import time # To measure training time

# Safety first
INPUT_CROP_SIZE : tuple[int, int] = (224, 224) # Standard input size for many CNNs
IMAGE_PATH : str = r"D:\temp\allDogImages.npy"
LABEL_PATH : str = r"D:\temp\allDogLabels.npy"

if (not os.path.exists(IMAGE_PATH) or not os.path.exists(LABEL_PATH)):
    print("Required data files not found. Please ensure the paths are correct.")
    exit(1)

# Load data
print("Loading data...")
images : np.ndarray = np.load(IMAGE_PATH)  # shape: (N, H, W, C)
labels : np.ndarray = np.load(LABEL_PATH)

#open the encoded file instead of encoding it again & removed sklearn lib
if os.path.exists("D:/temp/label_encoder.pkl"):
    with open("D:/temp/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully.")
else:
    print("Error: Label encoder not found at D:/temp/label_encoder.pkl.")
    sys.exit(1)
    
encoded_labels = label_encoder.transform(labels)
number_of_classes : int = len(label_encoder.classes_)

# Basic random cropping and random flip
print("Initializing image input transformation for training...")
transforms_train = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.RandomResizedCrop(size=INPUT_CROP_SIZE, antialias=True),
    v2.RandomHorizontalFlip(p=0.5), # Common horizontal flip augmentation
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Random color variation
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#This is for evaluation, so we use a center crop
print("Initializing image input transformation for evaluating...")
transforms_eval = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.CenterCrop(size=INPUT_CROP_SIZE),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset class using PIL
class DogBreedDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images.astype(np.uint8)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.images):
            raise IndexError("Index out of bounds for dataset.")
        
        img = self.images[idx]
        label = self.labels[idx]
        img = self.transform(img)
        return img, torch.tensor(label).long()

dataset = DogBreedDataset(images, encoded_labels, transforms_train)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) #increase batch size

# Load MobileNetV2
print("Initializing MobileNetV2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, number_of_classes)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Slightly lower LR for pretrained

# Training
print("Training...")
start = time.time() #start time for training
epoch_count : int = 20
  
for epoch in tqdm(range(epoch_count), desc="Training Epochs"):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epoch_count}] - Loss: {total_loss / len(dataloader):.4f}")

end = time.time() #end time for training
print(f"Training for {epoch_count} completed in {end - start:.2f} seconds.")

# Save model
print("Saving model...")
save_path = r"D:\temp\save model\dog_breed_mobilenetv2.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}. Success.")
