import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
import sys
import pickle
import numpy as np
from torchvision.transforms import v2  # New recommended version of torchvision transforms

from temperature_scaling import ModelWithTemperature # Import temperature scaling class

ENCODED_LABEL_PATH: str = r"D:\tsinghua_dataset_train\label_encoder.pkl"
IMAGE_VAL_PATH: str = r"D:\tsinghua_dataset_train\X_val.npy"
LABEL_VAL_PATH: str = r"D:\tsinghua_dataset_train\y_val.npy"

# Required files for temperature scaling
REQUIRED_FILES: list[str] = [ENCODED_LABEL_PATH, IMAGE_VAL_PATH, LABEL_VAL_PATH]
missingFiles = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missingFiles:
    print("Required validation files not found:")
    for f in missingFiles:
        print(f" - {f}")
    print("Please ensure the paths are correct.")
    sys.exit(1)

# === Load validation data ===
print("Loading validation images...")
valImages: np.ndarray = np.load(IMAGE_VAL_PATH)
valImages = valImages[..., ::-1].copy()  # BGR to RGB if needed

print("Loading validation labels...")
valLabels: np.ndarray = np.load(LABEL_VAL_PATH)

# === Load label encoder ===
print("Loading label encoder...")
with open(ENCODED_LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

encoded_val_labels = label_encoder.transform(valLabels)
number_of_classes: int = len(label_encoder.classes_)
print(f"Loaded label encoder with {number_of_classes} classes.")

print("Initializing image input transformation for validating...")
transforms_val = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.Resize(size=(256, 256), antialias=True), # Resize to a larger size before cropping
    v2.CenterCrop(size=(224, 224)),
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

print("Initializing DataLoader for validation...")
valDataset = DogBreedDataset(valImages, encoded_val_labels, transforms_val)
valDataloader = DataLoader(valDataset, batch_size=32, shuffle=False)

#Start the work
MODEL_PATH : str = r"D:\tsinghua_dataset_train\save model\dog_breed_mobilenetv2.pth" # Path to the trained model
CALIBRATED_MODEL_PATH: str = r"D:\tsinghua_dataset_train\save model\dog_breed_mobilenetv2_calibrated.pth"

# === Load original model ===
print("Loading MobileNetV2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

orig_model = models.mobilenet_v2(weights=None)
orig_model.classifier[1] = nn.Linear(orig_model.last_channel, number_of_classes)
orig_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
orig_model = orig_model.to(device)
orig_model.eval()

# === Apply temperature scaling ===
print("Calibrating with temperature scaling...")
scaled_model = ModelWithTemperature(orig_model).to(device)
scaled_model.set_temperature(valDataloader)

print(f"Optimal temperature found: {scaled_model.temperature.item():.4f}")

# === Save the calibrated model, not just state_dict ===
torch.save(scaled_model.state_dict(), CALIBRATED_MODEL_PATH)
print(f"Calibrated model saved to: {CALIBRATED_MODEL_PATH}")