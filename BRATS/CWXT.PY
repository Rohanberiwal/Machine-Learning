import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
import zipfile
import os

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

extract_brats_goat_data()
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

DATA_ROOT = r"C:\Users\rohan\OneDrive\Desktop\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth-003\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BratsGoATDataset(Dataset):
    def __init__(self, subject_paths, modalities=['t1c', 't1n', 't2w', 't2f'], transform=None):
        self.subject_paths = subject_paths
        self.modalities = modalities
        self.transform = transform

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        subject_path = self.subject_paths[idx]
        subject_id = os.path.basename(subject_path)
        
        images = []
        for mod in self.modalities:
            img_path = os.path.join(subject_path, f"{subject_id}-{mod}.nii.gz")
            img = nib.load(img_path).get_fdata()
            mid_slice = img.shape[2] // 2
            slice_2d = img[:, :, mid_slice]
            slice_2d = resize(slice_2d, (IMAGE_SIZE, IMAGE_SIZE), mode='constant', preserve_range=True)
            # Normalize slice between 0 and 1
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            images.append(slice_2d)
        image = np.stack(images, axis=0).astype(np.float32)

        seg_path = os.path.join(subject_path, f"{subject_id}-seg.nii.gz")
        seg = nib.load(seg_path).get_fdata()
        seg_2d = seg[:, :, seg.shape[2] // 2]
        seg_2d = resize(seg_2d, (IMAGE_SIZE, IMAGE_SIZE), order=0, preserve_range=True).astype(np.int64)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(seg_2d, dtype=torch.long)

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

def get_all_subjects(root):
    return [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm

BATCH_SIZE = 8
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_all_subjects(root):
    return [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

BATCH_SIZE = 8
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_all_subjects(root):
    return [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

class CvTWrapper(nn.Module):
    def __init__(self, num_classes=4, in_chans=4):
        super().__init__()
        self.backbone = timm.create_model(
            'cvt-13',  # fixed model name
            pretrained=False,
            in_chans=in_chans,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)


def train():
    subjects = get_all_subjects(DATA_ROOT)
    train_sub, val_sub = train_test_split(subjects, test_size=0.2, random_state=42)
    train_dataset = BratsGoATDataset(train_sub)
    val_dataset = BratsGoATDataset(val_sub)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = CvTWrapper(num_classes=4, in_chans=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "brats_cvt_model.pth")
    print("Model saved to brats_cvt_model.pth")

if __name__ == "__main__":
    train()
