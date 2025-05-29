#Operation Eldhrimnir: Stage 2.5
#Fine-tuning the model
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
import pickle
import os

print("Loading data and config...")
# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 #32
NUM_EPOCHS = 2 #20
LEARNING_RATE = 1e-4
MODEL_DIR = r"D:\temp\save model\dog_breed_mobilenetv2_finetuned.pth"
LABEL_ENCODER_DIR = r"D:\temp\label_encoder.pkl"

# Load data
X_train = np.load(r"D:\temp\X_train.npy")  # Shape: [N, H, W, C]
y_train = np.load(r"D:\temp\y_train.npy")
X_val = np.load(r"D:\temp\X_val.npy")
y_val = np.load(r"D:\temp\y_val.npy")

# Load label encoder to determine number of classes
with open(LABEL_ENCODER_DIR, "rb") as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

# Convert string labels to integer indices
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)

# Preprocess
X_train = torch.tensor(X_train).permute(0, 3, 1, 2).float() / 255.0
X_val = torch.tensor(X_val).permute(0, 3, 1, 2).float() / 255.0
y_train = torch.tensor(y_train).long()
y_val = torch.tensor(y_val).long()

# Define transformations
transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

X_train = transform(X_train)
X_val = transform(X_val)


# Define the model architecture
print("Loading model...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Load your previously trained weights
model.load_state_dict(torch.load(MODEL_DIR, map_location=device))
model = model.to(device)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Training loop
print("Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, "
          f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save the updated model
torch.save(model.state_dict(), r"D:\temp\save model\dog_breed_mobilenetv2_finetuned.pth")
print("Saving model to:", os.path.abspath("dog_breed_mobilenetv2_finetuned.pth"))
