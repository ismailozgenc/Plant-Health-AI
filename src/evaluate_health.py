import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from data_loader import PlantDataset, transform_species
from models.cnn_species_model import SpeciesCNN
from sklearn.metrics import accuracy_score

def evaluate_health_model(
    data_root: str = "Data",
    batch_size: int = 32,
    test_split: float = 0.2,
    model_load_path: str = "saved_models/health_model.pth"
):
    infer_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    infer_ds = PlantDataset(root_dir=data_root, mode='species', transform=infer_transform)
    infer_ld = DataLoader(infer_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    species_model = SpeciesCNN(num_classes=len(infer_ds.species_list))
    species_model.load_state_dict(torch.load(model_load_path.replace('health_model','species_model'), map_location=device))
    species_model.to(device).eval()

    correct_paths = []
    offset = 0
    with torch.no_grad():
        for imgs, labels in infer_ld:
            imgs = imgs.to(device)
            preds = torch.argmax(species_model(imgs), dim=1).cpu()
            for i in range(preds.size(0)):
                if preds[i] == labels[i]:
                    correct_paths.append(infer_ds.image_paths[offset + i])
            offset += preds.size(0)

    full_health = PlantDataset(root_dir=data_root, mode='health', transform=transform_species)
    valid_idx = [i for i, p in enumerate(full_health.image_paths) if p in correct_paths]
    health_ds = Subset(full_health, valid_idx)

    test_size = int(test_split * len(health_ds))
    train_size = len(health_ds) - test_size
    _, test_ds = random_split(health_ds, [train_size, test_size])

    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SpeciesCNN(num_classes=2)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.to(device).eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_ld:
            imgs = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred = np.hstack(all_preds)
    y_true = np.hstack(all_labels)

    acc = accuracy_score(y_true, y_pred) * 100
    print(f"Health model accuracy on {len(y_true)} samples: {acc:.2f}%")

if __name__ == '__main__':
    evaluate_health_model()
