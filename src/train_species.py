import os
import torch
import os
from torch.utils.data import DataLoader, random_split
from data_loader import PlantDataset
from models.cnn_species_model import SpeciesCNN # Our CNN Model
from utils import save_model

def train_species(
    data_root: str = "Data",
    species_list=None,
    batch_size: int = 32,
    val_split: float = 0.2,
    epochs: int = 40, 
    lr: float = 1e-4,
    model_save_path: str = "saved_models/species_model.pth"
):
    # 1) Dataset
    dataset = PlantDataset(
        root_dir=data_root,
        species_list=species_list,
        mode='species'
    )

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)
    num_classes = len(species_list) if species_list else len(dataset.species_list)
    model = SpeciesCNN(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100

        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Save best model for later stages/steps
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, model_save_path)

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    # Defined species
    species_list = ['Apple', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']

    train_species(
        data_root=os.path.join(os.getcwd(), 'Data'),
        species_list=species_list
    )