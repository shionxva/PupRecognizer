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

def sanity_check(dataloader : DataLoader, rows : int, cols : int) -> None:
    imgs, labels = next(iter(dataloader))  # imgs shape: [batch_size, 3, H, W]

    batch_size, _, H, W = imgs.shape
    n_show = min(batch_size, rows * cols)

    # Undo normalization and prepare images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    imgs_np = imgs[:n_show].cpu().permute(0, 2, 3, 1).numpy()  # [N, H, W, C]

    imgs_np = std * imgs_np + mean
    imgs_np = np.clip(imgs_np, 0, 1)
    imgs_np = (imgs_np * 255).astype(np.uint8)  # to uint8

    # Convert RGB to BGR for OpenCV
    imgs_np = imgs_np[..., ::-1]

    # Create a blank canvas to hold the grid
    grid_img = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

    for idx in range(n_show):
        row = idx // cols
        col = idx % cols
        grid_img[row*H:(row+1)*H, col*W:(col+1)*W, :] = imgs_np[idx]

    cv2.imshow(f"Batch of {n_show} images", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
trainImages: np.ndarray = np.load(IMAGE_TRAIN_PATH)
trainLabels: np.ndarray = np.load(LABEL_TRAIN_PATH)

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
number_of_classes : int = len(label_encoder.classes_)

# Basic random cropping and random flip
print("Initializing image input transformation for training...")
transforms_train = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
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
trainDataset = DogBreedDataset(trainImages, trainLabels, transforms_train)
trainDataloader = DataLoader(trainDataset, batch_size=32, shuffle=True)

print("Initializing DataLoader for validation...")
valDataset = DogBreedDataset(valImages, valLabels, transforms_val)
valDataloader = DataLoader(valDataset, batch_size=32, shuffle=False)

print("Sanity check for DataLoader...")
sanity_check(trainDataloader, 4, 8)  # Show 4 rows and 8 columns of images

print("Sanity check for validation DataLoader...")
sanity_check(valDataloader, 4, 8)

# Load MobileNetV2
print("Initializing MobileNetV2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, number_of_classes)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7)  # I AM SCARED OF THIS LEARNING RATE

# Training loop
print("Training...")
start = time.time() #start time for training
epoch_count : int = 20
  
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

    print(f"Epoch [{epoch + 1}/{epoch_count}] "
          f"- Train Loss: {avg_train_loss:.4f} "
          f"- Val Loss: {avg_val_loss:.4f} "
          f"- Val Acc: {val_accuracy:.2f}%")

end = time.time() #end time for training
print(f"Training for {epoch_count} completed in {end - start:.2f} seconds.")

# Save model
print("Saving model...")
save_path = r"D:\temp\save model\dog_breed_mobilenetv2.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}. Success.")
