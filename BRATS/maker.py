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

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = CBR(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

def train():
    subjects = get_all_subjects(DATA_ROOT)
    train_sub, val_sub = train_test_split(subjects, test_size=0.2, random_state=42)
    train_dataset = BratsGoATDataset(train_sub)
    val_dataset = BratsGoATDataset(val_sub)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "brats_unet_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()


extract_brats_goat_data()
train()