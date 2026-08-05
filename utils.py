import io
from PIL import Image
from torchvision import transforms

# Transforming the image to match the trained model's expected input format (224x224, normalized)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)