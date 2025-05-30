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
import pandas as pd # For reading CSV files
import cv2
import matplotlib.pyplot as plt # For visualizing images

def denormalize_image(tensor_img):
    """Reverses normalization for display purposes."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = tensor_img.clone().detach()
    img = img * std[:, None, None] + mean[:, None, None]
    img = img.permute(1, 2, 0).numpy()  # CHW to HWC
    img = np.clip(img, 0, 1)
    return img

def sanity_check_dataloader(dataloader, label_encoder, num_images=5) -> None:
    """Displays a few images and their decoded labels from a DataLoader."""
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        img = denormalize_image(images[i])
        label = label_encoder.inverse_transform([labels[i].item()])[0]

        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_metrics(train_losses, val_losses, val_accuracies) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='orange')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.plot(epochs, val_accuracies, label='Val Accuracy', color='green')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.title('Training & Validation Loss + Accuracy Over Epochs')
    plt.tight_layout()
    plt.show()

# Safety first
INPUT_CROP_SIZE : tuple[int, int] = (224, 224) # Standard input size for many CNNs
ENCODED_LABEL_PATH : str = r"D:/temp/label_encoder.pkl" # Path to the label file
IMAGE_TRAIN_PATH : str = r"D:\temp\X_train.npy"
LABEL_TRAIN_PATH : str = r"D:\temp\y_train.npy"
IMAGE_VAL_PATH : str = r"D:\temp\X_val.npy"
LABEL_VAL_PATH : str = r"D:\temp\y_val.npy"

#Error handling for missing files
REQUIRED_FILES : list[str] = [IMAGE_TRAIN_PATH, LABEL_TRAIN_PATH, IMAGE_VAL_PATH, LABEL_VAL_PATH]

missingFiles = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missingFiles:
    print("Required data files not found:")
    for f in missingFiles:
        print(f" - {f}")
    print("Please ensure the paths are correct.")
    exit(1)

# Load data
print("Loading all labels...")
df = pd.read_csv(r"D:\temp\labels.csv")
allLabels : np.ndarray = df['breed'].to_numpy()

print("Loading training data...")
trainImages : np.ndarray = np.load(IMAGE_TRAIN_PATH)
trainLabels : np.ndarray = np.load(LABEL_TRAIN_PATH)

trainImages = trainImages[..., ::-1].copy() #Convert BGR to RGB

print("Loading validation data...")
valImages: np.ndarray = np.load(IMAGE_VAL_PATH)
valLabels: np.ndarray = np.load(LABEL_VAL_PATH)

valImages = valImages[..., ::-1].copy() #Convert BGR to RGB

#open the encoded file instead of encoding it again & removed sklearn lib
if os.path.exists(ENCODED_LABEL_PATH):
    with open(ENCODED_LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully.")
else:
    print(f"Error: Label encoder not found at {ENCODED_LABEL_PATH}.")
    sys.exit(1)
    
encoded_labels = label_encoder.transform(allLabels)
encoded_train_labels = label_encoder.transform(trainLabels)
encoded_val_labels = label_encoder.transform(valLabels)
number_of_classes : int = len(label_encoder.classes_)

# Basic random cropping and random flip
print("Initializing image input transformation for training...")
transforms_train = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)), # Random Gaussian blur
    v2.RandomHorizontalFlip(p=0.5), # Common horizontal flip augmentation
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Random color variation
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Initializing image input transformation for validating...")
transforms_val = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
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

print("Initializing DataLoader for training...")
trainDataset = DogBreedDataset(trainImages, encoded_train_labels, transforms_train)
trainDataloader = DataLoader(trainDataset, batch_size=32, shuffle=True)

print("Initializing DataLoader for validation...")
valDataset = DogBreedDataset(valImages, encoded_val_labels, transforms_val)
valDataloader = DataLoader(valDataset, batch_size=32, shuffle=False)

print("Sanity check for DataLoader...")
sanity_check_dataloader(trainDataloader, label_encoder, num_images=5)

print("Sanity check for validation DataLoader...")
sanity_check_dataloader(valDataloader, label_encoder, num_images=5)

# Load MobileNetV2
print("Initializing MobileNetV2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, number_of_classes)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # I AM SCARED OF THIS LEARNING RATE

# Training loop
print("Training...")
start = time.time() #start time for training
epoch_count : int = 25

train_losses : list[float] = []
val_losses : list[float] = []
val_accuracies : list[float] = []

for epoch in tqdm(range(epoch_count), desc="Training Epochs"):
    # -- Training phase --
    model.train()
    total_loss = 0
    for images, targets in trainDataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss : float = total_loss / len(trainDataloader)

    # -- Validation phase --
    model.eval()
    val_loss : float = 0
    correct_predictions : int = 0
    total : int = 0

    with torch.no_grad():
        for images, targets in valDataloader:
            images, targets = images.to(device), targets.to(device)

            # Normal predict and calculate loss
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Get the index of the max log-probability
            predicted = torch.max(outputs, 1)[1]  

            #Sum boolean values in tensors to get the number of correct predictions
            correct_predictions += (predicted == targets).sum().item()
            total += targets.size(0)
    
    avg_val_loss : float = val_loss / len(valDataloader)
    val_accuracy : float = correct_predictions / total * 100

    # -- Append to plot later --
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{epoch_count}] "
          f"- Train Loss: {avg_train_loss:.4f} "
          f"- Val Loss: {avg_val_loss:.4f} "
          f"- Val Acc: {val_accuracy:.2f}%")

end = time.time() #end time for training
print(f"Training for {epoch_count} completed in {end - start:.2f} seconds.")

print("Plotting training metrics...")
plot_metrics(train_losses, val_losses, val_accuracies)

# Save model
print("Saving model...")
save_path = r"D:\temp\save model\dog_breed_mobilenetv2.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}. Success.")
