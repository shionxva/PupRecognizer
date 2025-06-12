import torch
import torch.nn as nn
from torchvision import models
import os

from temperature_scaling import ModelWithTemperature # Import temperature scaling class

CALIBRATED_MODEL_PATH : str = r"D:\tsinghua_dataset_train\save model\dog_breed_mobilenetv2_calibrated.pth"
num_classes : int = 130

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

dummy_input = torch.randn(1, 3, 224, 224, device=device)

torch_script_module = torch.jit.trace(calibrated_model, dummy_input)

# Save model
print("Saving model...")
torch_script_module.save(r"dog_breed_mobilenetv2_tsinghua_calibrated_torchscript.pt")