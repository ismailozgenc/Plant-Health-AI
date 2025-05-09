from utils import save_model
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import matplotlib.pyplot as plt

from data_loader import PlantDataset, transform_species
from utils import load_model
from models.cnn_species_model import SpeciesCNN
from models.get_knn import get_knn
from models.get_rf import get_rf
from models.get_svm import get_svm
from models.get_lr import get_lr
from sklearn.metrics import accuracy_score, confusion_matrix

def flatten_loader(loader):
    X, y = [], []
    for imgs, lbls in loader:
        arr = imgs.view(imgs.size(0), -1).numpy()
        X.append(arr)
        y.append(lbls.numpy())
    return np.vstack(X), np.hstack(y)


def train_and_evaluate():
    # Prepare species inference loader
    infer_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    infer_ds = PlantDataset(root_dir='Data', mode='species', transform=infer_transform)
    infer_ld = DataLoader(infer_ds, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)
    species_model = load_model(SpeciesCNN, 'saved_models/species_model.pth',
                               num_classes=len(infer_ds.species_list))
    species_model.to(device).eval()

    correct_paths = []
    offset = 0
    with torch.no_grad():
        for imgs, labels in infer_ld:
            bs = imgs.size(0)
            imgs = imgs.to(device)
            preds = torch.argmax(species_model(imgs), dim=1).cpu().numpy()
            true = labels.numpy()
            for i in range(bs):
                if preds[i] == true[i]:
                    correct_paths.append(infer_ds.image_paths[offset + i])
            offset += bs

    # Filtering data to select the images correctly identified
    full_health_ds = PlantDataset(root_dir='Data', mode='health', transform=transform_species)
    valid_idx = [i for i, p in enumerate(full_health_ds.image_paths) if p in correct_paths]
    ds = Subset(full_health_ds, valid_idx)

    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])
    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    X_train, y_train = flatten_loader(train_ld)
    X_test,  y_test  = flatten_loader(test_ld)

    # Classical models
    models = {
        'KNN': get_knn(),
        'RF':  get_rf(),
        'SVM': get_svm(),
        'LR':  get_lr()
    }
    import time
    results, times = {}, {}
    for name, mdl in models.items():
        start = time.time()
        mdl.fit(X_train, y_train)
        acc = accuracy_score(y_test, mdl.predict(X_test)) * 100
        times[name] = time.time() - start
        results[name] = acc

    # CNN for health (same architecture with species model since it was working well on the data)
    start_cnn = time.time()
    cnn = SpeciesCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        cnn.train()
        for imgs, lbls in train_ld:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(cnn(imgs), lbls)
            loss.backward()
            optimizer.step()
    cnn.eval()

    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls in test_ld:
            imgs = imgs.to(device)
            p = torch.argmax(cnn(imgs), dim=1).cpu().numpy()
            preds.append(p)
            trues.append(lbls.numpy())
    preds = np.hstack(preds)
    trues = np.hstack(trues)
    results['CNN'] = accuracy_score(trues, preds) * 100
    times['CNN'] = time.time() - start_cnn

    model_save_path = "saved_models/health_model.pth"
    save_model(cnn, model_save_path) # Saving model for inference module

    t_names, t_vals = zip(*times.items())
    plt.figure(figsize=(6,4))
    plt.bar(t_names, t_vals)
    plt.ylabel('Time (s)')
    plt.title('Method Execution Time')
    plt.tight_layout()
    plt.savefig('results/health_method_times.png')

    # Accuracy comparison bar plot
    names, accs = zip(*results.items())
    plt.figure(figsize=(6, 4))
    plt.bar(names, accs)
    plt.ylabel('Accuracy (%)')
    plt.title('Health Classification Comparison')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/health_model_comparison.png')

    # Cms per species for each model
    full_paths = [full_health_ds.image_paths[i] for i in test_ds.indices]
    species_list = full_health_ds.species_list

    all_preds = {name: mdl.predict(X_test) for name, mdl in models.items()}
    all_preds['CNN'] = preds

    for name, y_pred in all_preds.items():
        for sp in species_list:
            idxs = [i for i, p in enumerate(full_paths) if os.path.sep + sp + os.path.sep in p]
            if not idxs:
                continue

            y_true_sp = y_test[idxs]
            y_pred_sp = y_pred[idxs]
            cm = confusion_matrix(y_true_sp, y_pred_sp)

            plt.figure(figsize=(4,4))
            plt.imshow(cm, interpolation='nearest')
            plt.title(f'{name} — {sp}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks([0,1], ['Healthy','Unhealthy'])
            plt.yticks([0,1], ['Healthy','Unhealthy'])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i,j], ha='center', va='center')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'results/cm_{name}_{sp}.png')
            plt.close()

if __name__ == '__main__':
    train_and_evaluate()
