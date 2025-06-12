import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 # New recommended version of torchvision transforms
from torchvision import models
import os # For file handling and path management
from tqdm import tqdm # Progress bar for training epochs
import matplotlib.pyplot as plt # For visualizing images
import pickle # For loading and saving the label encoder
from PIL import Image # For image processing

from temperature_scaling import ModelWithTemperature # Import temperature scaling class

#Further labelled test
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

def sanity_check_image_paths(image_paths: list[str], test_type: str, num_images: int = 5) -> None:
    """Displays sample images using PIL and matplotlib, with a label indicating the test type."""
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(image_paths))):
        img_path = image_paths[i]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not open image at {img_path}. Error: {e}")
            continue

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")

    plt.suptitle(f"Sanity Check: Sample Images from {test_type} Test Set", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def load_images_from_directory(DIR: str, test_type : str) -> list[str]:
    """Loads all image file paths from a given directory."""
    print(f"Loading images from directory {DIR}")
    
    image_paths : list[str] = []

    for img_path in tqdm(os.listdir(DIR), total=len(os.listdir(DIR)), desc=f"Loading {test_type} test images"):
        full_img_path = os.path.join(DIR, img_path)
        image_paths.append(full_img_path)
    
    return image_paths

def plot_top_1_confidence_histogram(top_1_confidences: list[float], test_type: str) -> None:
    print("Plotting top-1 confidence histogram...")
    plt.figure(figsize=(8, 5))
    plt.hist(top_1_confidences, bins=20, color='mediumseagreen', edgecolor='black')
    plt.title(f"Histogram of Top-1 Confidence Scores ({test_type})")
    plt.xlabel("Top-1 Confidence (%)")
    plt.ylabel("Number of Predictions")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_average_confidence_per_rank(rank_confidences: list[list[float]], test_type: str) -> None:
    print("Plotting top-5 confidence scores...")

    # Compute average confidence for each rank
    avg_conf_per_rank = [sum(rank)/len(rank) for rank in rank_confidences]

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 6), avg_conf_per_rank, marker='o', color='dodgerblue')
    plt.title(f"Average Confidence from Top-1 to Top-5 ({test_type})") 
    plt.xlabel("Rank (1 = Top-1, 5 = Top-5)")
    plt.ylabel("Average Confidence (%)")
    plt.xticks(range(1, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_average_running_top_1_confidence(top_1_confidences: list[float], window_size: int, test_type: str) -> None:
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
    plt.title(f"Running Average of Top-1 Confidence Every {window_size} Images ({test_type})")
    plt.xlabel("Number of Images Processed")
    plt.ylabel("Average Top-1 Confidence (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def evaluate_test_images_confidence_only(image_paths: list[str], test_type: str, window_size : int, model: nn.Module, label_encoder, transforms_eval, device) -> None:
    print(f"Running prediction for {test_type} test...")
    top_1_confidences : list[float] = []
    rank_confidences : list[list[float]] = [[], [], [], [], []]
    
    for img_path in tqdm(image_paths, total=len(image_paths), desc=f"Evaluating {test_type} images"):
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

    # Histogram of top-1 confidence
    plot_top_1_confidence_histogram(top_1_confidences, test_type)

    # Plotting top-5 confidence scores
    plot_average_confidence_per_rank(rank_confidences, test_type)

    # Plotting running average of top-1 confidence
    plot_average_running_top_1_confidence(top_1_confidences, window_size, test_type)

def evaluate_labelled_test_images(image_paths: list[str], test_type: str, window_size: int, model: nn.Module, label_encoder, transforms_eval, device) -> None:
    print(f"Running prediction for {test_type} test (labelled)...")

    y_true: list[str] = []
    y_pred: list[str] = []

    top_1_confidences: list[float] = []
    rank_confidences: list[list[float]] = [[], [], [], [], []]  # For top-1 to top-5

    for img_path in tqdm(image_paths, total=len(image_paths), desc=f"Evaluating {test_type} images"):
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = transforms_eval(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            top5_confidence, top5_indices = torch.topk(probabilities, 5, dim=1)

        top1_index = top5_indices[0][0].item()
        top5_conf_np = top5_confidence[0].cpu().numpy()

        # Save confidence values
        top_1_confidences.append(top5_conf_np[0] * 100)
        for i in range(5):
            rank_confidences[i].append(top5_conf_np[i] * 100)

        # Get true and predicted labels
        predicted_class = label_encoder.inverse_transform([top1_index])[0]
        true_class = os.path.basename(os.path.dirname(img_path))

        y_pred.append(predicted_class)
        y_true.append(true_class)

    # Histogram of top-1 confidence
    plot_top_1_confidence_histogram(top_1_confidences, test_type)

    # Plotting top-5 confidence scores
    plot_average_confidence_per_rank(rank_confidences, test_type)

    # Plotting running average of top-1 confidence
    plot_average_running_top_1_confidence(top_1_confidences, window_size, test_type)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTop-1 Accuracy on {test_type} test: {acc * 100:.2f}%")

    # Use only breeds seen in this test set for Confusion Matrix and Per-Class Accuracy
    unique_labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix ({test_type} test) [Seen Breeds Only]")
    plt.tight_layout()
    plt.show()

    # Per-Class Accuracy
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        total_counts[t] += 1
        if t == p:
            correct_counts[t] += 1

    class_names = sorted(total_counts.keys())
    class_accuracies = [correct_counts[c] / total_counts[c] for c in class_names]

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, class_accuracies, color="salmon")
    plt.title(f"Per-Class Accuracy ({test_type} test) [Seen Breeds Only]")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Safety first
IMAGE_TEST_KAGGLE_DIR : str = r"D:\model_test_set\kaggle test" # Kaggle test set, basically from Stanford dataset
IMAGE_TEST_LABELLED_DIR : str = r"D:\model_test_set\labelled" # Labelled images from many datasets
IMAGE_TEST_UNLABELLED_DIR : str = r"D:\model_test_set\unlabelled" # Online images
CALIBRATED_MODEL_PATH : str = r"D:\tsinghua_dataset_train\save model\dog_breed_mobilenetv2_calibrated.pth" # Path to the trained model

#Error handling for missing files
REQUIRED_FILES : list[str] = [IMAGE_TEST_KAGGLE_DIR, IMAGE_TEST_LABELLED_DIR, IMAGE_TEST_UNLABELLED_DIR]

missingFiles = [f for f in REQUIRED_FILES if not (os.path.exists(f) or os.path.isdir(f))]

if missingFiles:
    print("Required data files not found:")
    for f in missingFiles:
        print(f" - {f}")
    print("Please ensure the paths are correct.")
    exit(1)

#Load kaggle and unlabelled test images
ALL_IMAGE_KAGGLE_PATH : list[str] = load_images_from_directory(IMAGE_TEST_KAGGLE_DIR, "kaggle")
ALL_IMAGE_UNLABELLED_PATH : list[str] = load_images_from_directory(IMAGE_TEST_UNLABELLED_DIR, "unlabelled")

#Load labelled test images
ALL_IMAGE_LABELLED_PATH : list[str] = []
for folder_name in tqdm(os.listdir(IMAGE_TEST_LABELLED_DIR), total=len(os.listdir(IMAGE_TEST_LABELLED_DIR)), desc="Loading labelled test folders"):
    folder_path = os.path.join(IMAGE_TEST_LABELLED_DIR, folder_name)
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_img_path = os.path.join(folder_path, img_name)
            ALL_IMAGE_LABELLED_PATH.append(full_img_path)

print("Confirming test sizes:")
print(f"Kaggle test images: {len(ALL_IMAGE_KAGGLE_PATH)}")
print(f"Unlabelled test images: {len(ALL_IMAGE_UNLABELLED_PATH)}") 
print(f"Labelled test images: {len(ALL_IMAGE_LABELLED_PATH)}")

print("Loading label encoder files...")
with open(r"D:/tsinghua_dataset_train/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

# Load the calibrated model
print(f"Loading calibrated MobileNetV2 model at {CALIBRATED_MODEL_PATH}...")
# Recreate the base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.mobilenet_v2(weights=None)
base_model.classifier[1] = nn.Linear(base_model.last_channel, 130)

# Wrap it again
calibrated_model = ModelWithTemperature(base_model).to(device)

# Load state dict
calibrated_model.load_state_dict(torch.load(CALIBRATED_MODEL_PATH, map_location=device))
calibrated_model.eval()

"""
#Load the trained model
print(f"Loading local MobileNetV2 model at {MODEL_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
"""

print("Initializing image input transformation for validating...")
transforms_eval = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.Resize(size=(256, 256), antialias=True),  # Resize to 256x256
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Sanity check image paths
"""
sanity_check_image_paths(ALL_IMAGE_KAGGLE_PATH, "Kaggle")
sanity_check_image_paths(ALL_IMAGE_UNLABELLED_PATH, "unlabelled")
sanity_check_image_paths(ALL_IMAGE_LABELLED_PATH, "labelled")
"""

evaluate_test_images_confidence_only(ALL_IMAGE_KAGGLE_PATH, "Kaggle", 200, calibrated_model, label_encoder, transforms_eval, device)
evaluate_test_images_confidence_only(ALL_IMAGE_UNLABELLED_PATH, "unlabelled", 10, calibrated_model, label_encoder, transforms_eval, device)

# Evaluate labelled test images
evaluate_labelled_test_images(ALL_IMAGE_LABELLED_PATH, "labelled", 100, calibrated_model, label_encoder, transforms_eval, device)