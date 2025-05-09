from data_loader import PlantDataset
from models.cnn_species_model import SpeciesCNN
from utils import load_model

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def visualize_species_performance(
    data_root: str = "Data",
    species_list=None,
    model_path: str = "saved_models/species_model.pth",
    batch_size: int = 32,
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)

    # (no augmentations)
    from torchvision import transforms
    transform_infer = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = PlantDataset(
        root_dir=data_root,
        species_list=species_list,
        mode='species',
        transform=transform_infer
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # trained model load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)
    num_classes = len(species_list) if species_list else len(dataset.species_list)
    model = load_model(SpeciesCNN, model_path, num_classes=num_classes)
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prdicted = np.array(all_predictions)

    total_accuracy = np.mean(y_prdicted == y_true) * 100

    # Per class accuracy
    per_class_acc = []
    for cls in range(len(species_list)):
        idxs = np.where(y_true == cls)[0]
        acc = np.mean(y_prdicted[idxs] == cls) * 100
        per_class_acc.append(acc)

    plt.figure(figsize=(10,5))
    sns.barplot(x=species_list, y=per_class_acc)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Species')
    plt.title('Per-Class Accuracy for Species Classification')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))
    plt.close()

    cm = confusion_matrix(y_true, y_prdicted)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=species_list,
                yticklabels=species_list,
                cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Species Classification Confusion Matrix Total Accuracy: {total_accuracy:.1f}%')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"Saved per-class accuracy and confusion matrix to '{output_dir}/'")

if __name__ == '__main__':
    species_list = ['Apple', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']

    visualize_species_performance(
        data_root=os.path.join(os.getcwd(), 'Data'),
        species_list=species_list
    )
