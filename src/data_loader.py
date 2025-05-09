import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Transforms of Augmented RGB
transform_species = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_health = transform_species # it is goigng to be the same transform for 2 modes

class PlantDataset(Dataset):
    def __init__(self, root_dir, species_list=None, mode='species', transform=None):
        self.root_dir = root_dir
        self.mode = mode  # 'species' or 'health'
        self.species_list = (species_list if species_list 
                             else sorted([d for d in os.listdir(root_dir) 
                                          if os.path.isdir(os.path.join(root_dir, d))]))
        self.transform = transform or (transform_species if mode=='species' else transform_health)

        self.image_paths = []
        self.labels = []
        for idx, species in enumerate(self.species_list):
            species_dir = os.path.join(root_dir, species)
            for health_idx, health in enumerate(['Healthy', 'Unhealthy']):
                folder = os.path.join(species_dir, health)
                for fname in os.listdir(folder):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(folder, fname))
                        if mode == 'species':
                            self.labels.append(idx)
                        else:
                            self.labels.append(health_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label