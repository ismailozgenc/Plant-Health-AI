'''
This code has been executed to make sure that different class sizes are equal and balanced.   
'''
import os, random
from PIL import Image
from torchvision import transforms

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.abspath(os.path.dirname(__file__))   # use the folder the script lives in
STRATEGY    = 'both'        # to undersample if oversized and oversample if otherwise
TARGET_MODE = 'max'         
TARGET_CUSTOM = 1000       
# ──────────────────────────────────────────────────────────────────────────────

# simple augmentation for oversampling
aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
])

# collect all species/status paths
folders = {}
for species in os.listdir(DATA_DIR):
    spath = os.path.join(DATA_DIR, species)
    if not os.path.isdir(spath): continue
    for status in ('healthy','unhealthy'):
        here = os.path.join(spath, status)
        if os.path.isdir(here):
            key = f"{species}/{status}"
            folders[key] = here

# count images before
before_counts = {k: len([f for f in os.listdir(p) 
                         if f.lower().endswith(('.jpg','.png','.jpeg'))]) 
                 for k,p in folders.items()}

# decide target
if TARGET_MODE == 'min':
    target = min(before_counts.values())
elif TARGET_MODE == 'max':
    target = max(before_counts.values())
else:
    target = TARGET_CUSTOM

print("Before balancing:", before_counts)
print(f"Target per folder: {target}\n")

# balancing phase
for key, path in folders.items():
    imgs = [f for f in os.listdir(path) 
            if f.lower().endswith(('.jpg','.png','.jpeg'))]

    # 1) undersample
    if STRATEGY in ('undersample','both') and len(imgs) > target:
        to_rm = random.sample(imgs, len(imgs) - target)
        for fn in to_rm:
            os.remove(os.path.join(path, fn))
        imgs = [f for f in imgs if f not in to_rm]

    # 2) oversample
    if STRATEGY in ('oversample','both') and len(imgs) < target:
        deficit = target - len(imgs)
        for i in range(deficit):
            src = random.choice(imgs)
            img = Image.open(os.path.join(path, src))
            img_aug = aug(img)
            base, ext = os.path.splitext(src)
            new_name = f"aug_{i}_{base}{ext}"
            img_aug.save(os.path.join(path, new_name))

# to be sure they are equal
after_counts = {k: len([f for f in os.listdir(p) 
                        if f.lower().endswith(('.jpg','.png','.jpeg'))]) 
                for k,p in folders.items()}
print("After balancing: ", after_counts)
