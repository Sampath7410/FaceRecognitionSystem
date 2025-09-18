import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19(weights="IMAGENET1K_V1")
model.classifier[6] = torch.nn.Linear(4096, len(class_names))
model.load_state_dict(torch.load("vgg19_face_recognition.pth", map_location=device))
model = model.to(device)
model.eval()

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[str(predicted.item())]

    return predicted_class
