#Operation Verthandi - Stage 2
#Create a trained model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from tqdm import tqdm
import pickle

# Load data
print("Loading data...")
images = np.load(r"D:\temp\allDogImages.npy")  # shape: (N, H, W, C)
labels = np.load(r"D:\temp\allDogLabels.npy")

#open the encoded file instead of encoding it again & removed sklearn lib
with open("D:/temp/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

labels_encoded = label_encoder.transform(labels)  # strings to integers
num_classes = len(label_encoder.classes_)

# MobileNetV2 transf (224x224 with ImageNet normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]) 
])

# Dataset
class DogBreedDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images.astype(np.uint8)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = self.transform(img)
        return img, torch.tensor(label).long()

dataset = DogBreedDataset(images, labels_encoded, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) #increase batch size


# Load MobileNetV2
print("Initializing MobileNetV2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Slightly lower LR for pretrained

# Training
print("Training...")
epoch_count = 20
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


# Save model
print("Saving model...")
save_path = r"D:\temp\save model\dog_breed_mobilenetv2.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}. Success.")
