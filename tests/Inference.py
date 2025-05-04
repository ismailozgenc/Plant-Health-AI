import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from utils import load_model
from models.cnn_species_model import SpeciesCNN  # Our CNN Model for Species
from models.cnn_health_model import HealthCNN    # Our CNN Model for Health

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

species_list = ['Apple', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']

def predict(image, species_model, health_model, transform, device):
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Species prediction with confidence
    species_logits = species_model(image_tensor)
    species_probs = torch.softmax(species_logits, dim=1)
    species_idx = species_probs.argmax(dim=1).item()
    species_confidence = species_probs[0, species_idx].item()

    # Health prediction with confidence
    health_logits = health_model(image_tensor)
    health_probs = torch.softmax(health_logits, dim=1)
    health_idx = health_probs.argmax(dim=1).item()
    health_confidence = health_probs[0, health_idx].item()

    species_name = species_list[species_idx]
    health_status = 'Healthy' if health_idx == 0 else 'Unhealthy'

    return species_name, health_status, species_confidence, health_confidence

def batch_inference(image_folder, species_model, health_model, transform, device):
    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_name)
            image = Image.open(image_path).convert('RGB')

            species, health, species_confidence, health_confidence = predict(
                image, species_model, health_model, transform, device
            )

            title = (
                f"{image_name}\n"
                f"Species: {species} ({species_confidence:.1%})   "
                f"Health: {health} ({health_confidence:.1%})"
            )

            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.title(title, fontsize=10)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    image_folder = "unlabeled_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    species_model = load_model(SpeciesCNN, 'saved_models/species_model.pth',
                               num_classes=8)
    species_model.to(device).eval()

    health_model = load_model(HealthCNN, "saved_models/health_model.pth")
    health_model.to(device).eval()

    # Run batch for inference
    batch_inference(image_folder, species_model, health_model, transform, device)
