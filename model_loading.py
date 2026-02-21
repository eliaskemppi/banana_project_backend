import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

# 1. Define your custom architecture again (must match your notebook)

class BananaNet(nn.Module):
    def __init__(self, num_classes=2):
        super(BananaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # 224 -> 112 -> 56 after 2 pools
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_models():
    # Load Custom CNN
    custom_model = BananaNet()
    custom_model.load_state_dict(torch.load("saved_models/BananaNet.pth", map_location="cpu"))
    custom_model.eval() # Set to evaluation mode

    # Load MobileNetV2
    mobilenet = models.mobilenet_v2()
    mobilenet.classifier[1] = nn.Linear(1280, 2) # Match your output classes
    mobilenet.load_state_dict(torch.load("saved_models/MobileNet.pth", map_location="cpu"))
    mobilenet.eval()

    return custom_model, mobilenet