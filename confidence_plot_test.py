import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 # New recommended version of torchvision transforms
from torchvision import models
import os # For file handling and path management
from tqdm import tqdm # Progress bar for training epochs
import sys # For early error handling
import matplotlib.pyplot as plt # For visualizing images
import pickle # For loading and saving the label encoder
from PIL import Image # For image processing

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

# Safety first
IMAGE_TEST_DIR : str = r"D:\temp\test"
ONLINE_IMAGE_TEST_DIR : str = r"D:\temp\online images"
#Error handling for missing files
REQUIRED_FILES : list[str] = [IMAGE_TEST_DIR, ONLINE_IMAGE_TEST_DIR]

missingFiles = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missingFiles:
    print("Required data files not found:")
    for f in missingFiles:
        print(f" - {f}")
    print("Please ensure the paths are correct.")
    exit(1)

print("Joining the offline and online image test paths...")
ALL_IMAGE_TEST_PATH : list[str] = []
for dir in tqdm([IMAGE_TEST_DIR, ONLINE_IMAGE_TEST_DIR], total=2, desc="Loading image paths"):
    if os.path.exists(dir) and os.path.isdir(dir):
        image_paths = [os.path.join(dir, path) for path in os.listdir(dir) if path.lower().endswith(('.png', '.jpg', '.jpeg'))]
        ALL_IMAGE_TEST_PATH.extend(image_paths)

print("Loading label encoder files...")
with open(r"D:/temp/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

#Load the trained model
print("Loading local MobileNetV2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(r"D:/temp/save model/dog_breed_mobilenetv2.pth", map_location=device))
model = model.to(device)
model.eval()

print("Initializing image input transformation for validating...")
transforms_eval = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inference
top_1_confidences : list[float] = []
rank_confidences : list[list[float]] = [[], [], [], [], []] # For top 5 confidences

print("Running prediction...")
for img_path in tqdm(ALL_IMAGE_TEST_PATH, desc="Evaluating images"):
    #Load and preprocess test image
    image = Image.open(img_path).convert("RGB")
    image = transforms_eval(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        top5_confidence, top5_indices = torch.topk(probabilities, 5, dim=1)

    # Convert to CPU + numpy
    top5_conf_np = top5_confidence[0].cpu().numpy()

    # Store top-1 confidence
    top_1_confidences.append(top5_conf_np[0] * 100)

    # Store each rank's confidence (0 = top-1, 4 = top-5)
    for i in range(5):
        rank_confidences[i].append(top5_conf_np[i] * 100)

#Histogram of top-1 confidence
print("Plotting top-1 confidence histogram...")
plt.figure(figsize=(8, 5))
plt.hist(top_1_confidences, bins=20, color='mediumseagreen', edgecolor='black')
plt.title("Histogram of Top-1 Confidence Scores")
plt.xlabel("Top-1 Confidence (%)")
plt.ylabel("Number of Predictions")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plotting top-5 confidence scores
print("Plotting top-5 confidence scores...")
# Compute average confidence for each rank
avg_conf_per_rank = [sum(rank)/len(rank) for rank in rank_confidences]

plt.figure(figsize=(7, 4))
plt.plot(range(1, 6), avg_conf_per_rank, marker='o', color='dodgerblue')
plt.title("Average Confidence per Rank (Top-5)")
plt.xlabel("Rank (1 = Top-1, 5 = Top-5)")
plt.ylabel("Average Confidence (%)")
plt.xticks(range(1, 6))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

window_size = 500
running_avgs = []

for i in range(0, len(top_1_confidences), window_size):
    window = top_1_confidences[i:i+window_size]
    if len(window) == 0:
        break
    running_avgs.append(sum(window) / len(window))

# x-axis points: number of images covered in each window
x_vals = [min(window_size * (i + 1), len(top_1_confidences)) for i in range(len(running_avgs))]
print("Plotting running average of top-1 confidence...")
plt.figure(figsize=(10, 5))
plt.plot(x_vals, running_avgs, marker='o', color='teal')
plt.title(f"Running Average of Top-1 Confidence Every {window_size} Images")
plt.xlabel("Number of Images Processed")
plt.ylabel("Average Top-1 Confidence (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()