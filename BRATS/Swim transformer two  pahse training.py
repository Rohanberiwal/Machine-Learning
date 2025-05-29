import os
import zipfile
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

def extract_brats_goat_data():
    zip_paths = [
        r"C:\Users\rohan\OneDrive\Desktop\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth-003.zip",
        r"C:\Users\rohan\OneDrive\Desktop\MICCAI2024-BraTS-GoAT-ValidationData-002.zip",
        r"C:\Users\rohan\OneDrive\Desktop\MICCAI2024-BraTS-GoAT-TrainingData-WithOut-GroundTruth-001.zip"
    ]
    for zip_path in zip_paths:
        if os.path.exists(zip_path):
            output_dir = zip_path.replace('.zip', '')
            os.makedirs(output_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

DATA_ROOT = r"C:\Users\rohan\OneDrive\Desktop\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth-003\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import shutil
import numpy as np
import nibabel as nib
from skimage.transform import resize, rotate
from collections import defaultdict
from tqdm import tqdm

# CONFIG
DATA_ROOT = r"C:\Users\rohan\OneDrive\Desktop\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth-003\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
SAVE_DIR = os.path.join(r"C:\Users\rohan\OneDrive\Desktop", "augmentation_balanced")
IMAGE_SIZE = 224
MODALITIES = ['t1c', 't1n', 't2w', 't2f']

import os
import shutil
import stat

SAVE_DIR = 'C:\\Users\\rohan\\OneDrive\\Desktop\\augmentation_balanced'

import os
import shutil
import stat

SAVE_DIR = 'C:\\Users\\rohan\\OneDrive\\Desktop\\augmentation_balanced'

def onerror(func, path, exc_info):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR, onerror=onerror)

os.makedirs(SAVE_DIR)


if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)


def get_all_subjects(root):
    return [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

def normalize_and_resize(img):
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return resize(img, (IMAGE_SIZE, IMAGE_SIZE), preserve_range=True)

def augment_image(image, mask):
    return [
        (np.flip(image, axis=2), np.flip(mask, axis=1)),
        (rotate(image, angle=15, mode='edge', preserve_range=True), rotate(mask, angle=15, order=0, mode='edge', preserve_range=True)),
        (rotate(image, angle=-15, mode='edge', preserve_range=True), rotate(mask, angle=-15, order=0, mode='edge', preserve_range=True))
    ]

def compute_class_distribution(subject_paths):
    counter = defaultdict(int)
    for subj_path in subject_paths:
        subject_id = os.path.basename(subj_path)
        seg_path = os.path.join(subj_path, f"{subject_id}-seg.nii.gz")
        if not os.path.exists(seg_path):
            print(f"[Warning] Missing segmentation file: {seg_path}")
            continue
        try:
            seg = nib.load(seg_path).get_fdata()
            mid = seg[:, :, seg.shape[2] // 2]
            mid_resized = resize(mid, (IMAGE_SIZE, IMAGE_SIZE), order=0, preserve_range=True).astype(np.int64)
            unique, counts = np.unique(mid_resized, return_counts=True)
            for u, c in zip(unique, counts):
                counter[u] += c
        except Exception as e:
            print(f"[Error] Could not process {seg_path}: {e}")
    return counter


subject_paths = get_all_subjects(DATA_ROOT)
print(f"Found {len(subject_paths)} subjects in {DATA_ROOT}")

if not subject_paths:
    raise ValueError(f"No subject folders found in {DATA_ROOT}")

original_distribution = compute_class_distribution(subject_paths)
print(f"Original class distribution: {original_distribution}")

if not original_distribution or all(v == 0 for v in original_distribution.values()):
    raise ValueError("No valid segmentation data found. Please verify the file paths and contents.")

max_pixels = max(original_distribution.values())
target_pixels = {cls: max_pixels for cls in original_distribution if cls != 0}
class_pixel_counter = defaultdict(int)
aug_counter = 0


augmentation_dict = {}

for subj_path in tqdm(subject_paths):
    subject_id = os.path.basename(subj_path)
    seg_path = os.path.join(subj_path, f"{subject_id}-seg.nii.gz")
    if not os.path.exists(seg_path):
        print(f"[Skip] No segmentation file for subject: {subject_id}")
        continue

    try:
        seg = nib.load(seg_path).get_fdata()
        seg_slice = seg[:, :, seg.shape[2] // 2]
        seg_resized = resize(seg_slice, (IMAGE_SIZE, IMAGE_SIZE), order=0, preserve_range=True).astype(np.int64)

        class_present = [cls for cls in np.unique(seg_resized) if cls in target_pixels and class_pixel_counter[cls] < target_pixels[cls]]
        if not class_present:
            continue

        img_slices = []
        missing_modality = False
        for mod in MODALITIES:
            img_path = os.path.join(subj_path, f"{subject_id}-{mod}.nii.gz")
            if not os.path.exists(img_path):
                print(f"[Skip] Missing modality file: {img_path}")
                missing_modality = True
                break
            img = nib.load(img_path).get_fdata()
            mid_slice = img[:, :, img.shape[2] // 2]
            img_resized = normalize_and_resize(np.nan_to_num(mid_slice))
            img_slices.append(img_resized)

        if missing_modality:
            continue

        img_stack = np.stack(img_slices)
        augmented = augment_image(img_stack, seg_resized)

        for aug_img_stack, aug_mask in augmented:
            aug_classes = np.unique(aug_mask)
            if not any(cls in target_pixels and class_pixel_counter[cls] < target_pixels[cls] for cls in aug_classes):
                continue

            folder_name = f"{subject_id}_aug_{aug_counter}"
            save_path = os.path.join(SAVE_DIR, folder_name)
            os.makedirs(save_path, exist_ok=True)

            entry = {}

            for m, mod in enumerate(MODALITIES):
                mod_path = os.path.join(save_path, f"{mod}.npy")
                np.save(mod_path, aug_img_stack[m])
                entry[mod] = mod_path

            seg_save_path = os.path.join(save_path, "seg.npy")
            np.save(seg_save_path, aug_mask)
            entry['seg'] = seg_save_path

            augmentation_dict[folder_name] = entry

            for cls in aug_classes:
                if cls in target_pixels:
                    class_pixel_counter[cls] += np.sum(aug_mask == cls)
            aug_counter += 1

            if all(class_pixel_counter[c] >= target_pixels[c] for c in target_pixels):
                print("All target pixel counts satisfied.")
                break

        if all(class_pixel_counter[c] >= target_pixels[c] for c in target_pixels):
            break

    except Exception as e:
        print(f"[Error] Problem processing subject {subject_id}: {e}")
        continue

print("\n Augmentation completed.")
print(f"Final class pixel count: {dict(class_pixel_counter)}")
print(f"Total augmented samples: {len(augmentation_dict)}")

import json
dict_path = os.path.join(SAVE_DIR, "augmentation_index.json")
with open(dict_path, "w") as f:
    json.dump(augmentation_dict, f, indent=2)

print(f"Augmentation dictionary saved to {dict_path}")

class BratsGoATDataset(Dataset):
    def __init__(self, subject_paths, modalities=['t1c', 't1n', 't2w', 't2f']):
        self.subject_paths = subject_paths
        self.modalities = modalities

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        subject_path = self.subject_paths[idx]
        subject_id = os.path.basename(subject_path)

        images = []
        for mod in self.modalities:
            img_path = os.path.join(subject_path, f"{subject_id}-{mod}.nii.gz")
            try:
                img = nib.load(img_path).get_fdata()
            except Exception:
                img = np.zeros((240, 240, 155))

            mid_slice = img.shape[2] // 2
            slice_2d = img[:, :, mid_slice]

            slice_2d = np.nan_to_num(slice_2d, nan=0.0)
            p1, p99 = np.percentile(slice_2d, (1, 99))
            slice_2d = np.clip(slice_2d, p1, p99)
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            slice_2d = resize(slice_2d, (IMAGE_SIZE, IMAGE_SIZE), mode='constant', preserve_range=True)
            images.append(slice_2d)

        image = np.stack(images, axis=0).astype(np.float32)

        seg_path = os.path.join(subject_path, f"{subject_id}-seg.nii.gz")
        try:
            seg = nib.load(seg_path).get_fdata()
        except Exception:
            seg = np.zeros((240, 240, 155))

        seg_2d = seg[:, :, seg.shape[2] // 2]
        seg_2d = resize(seg_2d, (IMAGE_SIZE, IMAGE_SIZE), order=0, preserve_range=True).astype(np.int64)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(seg_2d, dtype=torch.long)

def get_all_subjects(root):
    return [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class BratsGoATAugmentedDataset(Dataset):
    def __init__(self, augmentation_dir):
        self.augmentation_dir = augmentation_dir
        dict_path = os.path.join(augmentation_dir, "augmentation_index.json")
        with open(dict_path, "r") as f:
            self.augmentation_dict = json.load(f)
        self.samples = list(self.augmentation_dict.keys())
        self.modalities = ['t1c', 't1n', 't2w', 't2f']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_folder = self.samples[idx]
        sample_info = self.augmentation_dict[sample_folder]
        imgs = []
        for mod in self.modalities:
            mod_path = sample_info.get(mod)
            if mod_path is None:
                raise RuntimeError(f"Missing modality {mod} for sample {sample_folder}")
            img = np.load(mod_path)
            imgs.append(img)
        image = np.stack(imgs).astype(np.float32)
        seg_path = sample_info.get('seg')
        if seg_path is None:
            raise RuntimeError(f"Missing segmentation for sample {sample_folder}")
        seg = np.load(seg_path).astype(np.int64)
        image_tensor = torch.tensor(image, dtype=torch.float32)
        seg_tensor = torch.tensor(seg, dtype=torch.long)
        return image_tensor, seg_tensor
    
from torch.utils.data import DataLoader

augmented_dataset = BratsGoATAugmentedDataset(augmentation_dir="augmentation")
augmented_loader = DataLoader(augmented_dataset, batch_size=4, shuffle=True, num_workers=2)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNeXtWrapper(nn.Module):
    def __init__(self, num_classes=4, in_chans=4):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, in_chans=in_chans, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses

print("Training on augmented data...")

augmented_dataset = BratsGoATAugmentedDataset(SAVE_DIR)
train_size_aug = int(0.8 * len(augmented_dataset))
val_size_aug = len(augmented_dataset) - train_size_aug
train_set_aug, val_set_aug = random_split(augmented_dataset, [train_size_aug, val_size_aug])

train_loader_aug = DataLoader(train_set_aug, batch_size=8, shuffle=True, num_workers=4)
val_loader_aug = DataLoader(val_set_aug, batch_size=8, shuffle=False, num_workers=4)

model_aug = ConvNeXtWrapper(num_classes=4, in_chans=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_aug = optim.Adam(model_aug.parameters(), lr=1e-4)

train_losses_aug, val_losses_aug = train_model(model_aug, train_loader_aug, val_loader_aug, criterion, optimizer_aug, epochs=50)

plt.plot(train_losses_aug, label='Train Loss - Augmented (CVT)')
plt.plot(val_losses_aug, label='Val Loss - Augmented (CVT)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ConvNeXt Training - Augmented Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convnext_training_loss.png")
plt.show()
