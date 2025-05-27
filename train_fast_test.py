#Operation Verthandi - Stage 2
#Create a trained model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import os
import time
from tqdm import tqdm


# Load data
print("Loading data...")
images = np.load(r"D:\temp\allDogImages.npy")  # shape: (N, H, W, C)
labels = np.load(r"D:\temp\allDogLabels.npy")

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)  # strings to integers
num_classes = len(label_encoder.classes_)

# Resize to 64x64 and convert to tensor
print("Resizing images...")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # converts to [0, 1]
])

# Dataset using PIL
class DogBreedDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images.astype(np.uint8)  # ensure uint8
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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# test on Tiny CNN model
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Setup
print("Initiating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(labels))
model = TinyCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Fast training
print("Training...")
epoch_count : int = 10
for epoch in tqdm(range(epoch_count), desc="Training Epochs"):  # only 2 epochs for fast test
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
os.makedirs(r"D:\temp\save model", exist_ok=True)
torch.save(model.state_dict(), r"D:\temp\save modeldog_breed_tinycnn_fasttest.pth")
print("Model saved. Success.")
