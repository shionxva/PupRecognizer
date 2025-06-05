import torch
import torch.nn as nn
from torchvision import models
import os

MODEL_PATH : str = r"D:\stanford_dataset_train\save model\dog_breed_mobilenetv2.pth"
num_classes : int = 120

#Load the trained model
print(f"Loading local MobileNetV2 model at {MODEL_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, device=device)

torch_script_module = torch.jit.trace(model, dummy_input)

# Save model
print("Saving model...")
torch_script_module.save(r"dog_breed_mobilenetv2_torchscript.pt")