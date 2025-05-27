#Operation Eldhrimnir - Stage 3
#Run
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import os

#Define the model (same architecture)
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

#Load label encoder
print("Loading files...")
with open("D:/temp/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

#Load the trained model
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN(num_classes).to(device)
model.load_state_dict(torch.load("D:/temp/dog_breed_tinycnn_fasttest.pth", map_location=device))
model.eval()

#Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

#Load and preprocess test image
print("Inputing...")
img_path = "D:/test_subject3.jpg"  # Change this
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

#Inference
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

print(f"Predicted dog breed: {predicted_label}")
print("Success")