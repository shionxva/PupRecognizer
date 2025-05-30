#Operation Skuld - Stage 3
#Run
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
import pickle
import os


#Load label encoder
print("Loading files...")
with open(r"D:/temp/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

#Load the trained model
print("Loading MobileNetV2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(r"D:/temp/save model/dog_breed_mobilenetv2.pth", map_location=device))
model = model.to(device)
model.eval()

# Define preprocessing (use ImageNet standard normalization!)
#This is for evaluation, so we use a center crop
print("Initializing image input transformation for evaluating...")
transforms_eval = v2.Compose([
    v2.ToImage(), # Convert numpy array to tensor
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess test image
print("Inputing...")
img_path = r"D:\temp\test\00c14d34a725db12068402e4ce714d4c.jpg"  # Change this
image = Image.open(img_path).convert("RGB")
image = transforms_eval(image).unsqueeze(0).to(device)  # Add batch dimension

# Inference
print("Running prediction...")
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    top5_confidence, top5_indices = torch.topk(probabilities, 5, dim=1)

# Decode top 5 predicted labels and confidences
top5_labels = label_encoder.inverse_transform(top5_indices.cpu().numpy()[0])
top5_confidences = top5_confidence.cpu().numpy()[0] * 100

print("\nTop 5 Predictions:")
for i in range(5):
    print(f"{i+1}. {top5_labels[i]} - Confidence: {top5_confidences[i]:.2f}%")

print("Success")
