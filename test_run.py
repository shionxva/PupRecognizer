#Operation Eldhrimnir - Stage 3
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

#Load and preprocess test image
print("Inputing...")
img_path = r"D:\Downloads\pexels-hnoody93-58997.jpg"  # Change this
image = Image.open(img_path).convert("RGB")
image = transforms_eval(image).unsqueeze(0).to(device)  # Add batch dimension

# Inference
print("Running prediction...")
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
    confidence_percent = confidence.item() * 100

print(f"\nPredicted Dog Breed: {predicted_label}")
print(f"Confidence: {confidence_percent:.2f}%")
print("Success")
