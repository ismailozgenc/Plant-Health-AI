import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, r"C:\Users\ara_4\anaconda3\Lib\site-packages")
print('\n', sys.path, '\n')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
from utils import load_model
from models.cnn_species_model import SpeciesCNN  # Our CNN Model for Species
from models.cnn_health_model import HealthCNN    # Our CNN Model for Health
from torchvision import transforms

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

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Classifier")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root.geometry("600x600")

        self.species_model = load_model(SpeciesCNN, 'saved_models/species_model.pth', num_classes=8).to(self.device).eval()
        self.health_model = load_model(HealthCNN, "saved_models/health_model.pth").to(self.device).eval()

        self.background_image = Image.open("background.jpg")
        self.background_image = self.background_image.resize((600, 600)) 
        self.bg_image_tk = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(root, image=self.bg_image_tk)
        self.background_label.place(relwidth=1, relheight=1) 

        self.select_button = tk.Button(root, text="Select Image", command=self.load_image, font=("Arial", 12))
        self.select_button.place(x=250, y=20)  
        self.image_label = tk.Label(root, bg="white", relief="solid")
        self.image_label.place(x=170, y=100, width=256, height=256)  

        self.result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=480, justify="center", bg="white", relief="solid")
        self.result_label.place(x=60, y=400, width=480, height=100) 

        self.status_label = tk.Label(root, text="", font=("Arial", 10), fg="blue", bg="white", relief="solid")
        self.status_label.place(x=200, y=520, width=200, height=30)  

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.status_label.config(text="Evaluating...")
        self.root.update_idletasks()

        image = Image.open(file_path).convert("RGB")
        display_image = ImageTk.PhotoImage(image.resize((256, 256)))
        self.image_label.config(image=display_image)
        self.image_label.image = display_image

        #Prediction
        species, health, species_conf, health_conf = predict(image, self.species_model, self.health_model, transform, self.device)
        result_text = f"Species: {species} ({species_conf:.1%})\nHealth: {health} ({health_conf:.1%})"

        #Status
        self.result_label.config(text=result_text)
        self.status_label.config(text="Evaluation completed.")


root = tk.Tk()
app = PlantDiseaseApp(root)
root.mainloop()
