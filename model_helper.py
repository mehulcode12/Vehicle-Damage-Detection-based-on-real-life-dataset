from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True            
            
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

trained_model = None
class_names = ['Front_Breakage', 'Front_Crushed', 'Front_Normal', 'Rear_Breakage', 'Rear_Crushed', 'Rear_Normal']

def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension #IMPOrtant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    global trained_model
    # Check if the model is already loaded
    if trained_model is None:
        # Load the model
        trained_model = CarClassifierResNet(num_classes=6)  
        trained_model.load_state_dict(torch.load("model\saved_model.pth", map_location=device))
        trained_model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = trained_model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]


    return "Prediction result"