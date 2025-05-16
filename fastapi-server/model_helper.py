from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

# Define model architecture
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Global model and class names
trained_model = None
class_names = ['Front_Breakage', 'Front_Crushed', 'Front_Normal', 'Rear_Breakage', 'Rear_Crushed', 'Rear_Normal']

# ✅ Updated: Accepts a PIL image instead of path
def predict(image: Image.Image) -> str:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet(num_classes=6)
        trained_model.load_state_dict(torch.load("model/saved_model.pth", map_location=device))
        trained_model.eval()
        trained_model.to(device)

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = trained_model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]
