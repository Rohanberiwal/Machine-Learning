import os
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import numpy as np

input_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS Data\AIIMS Data"
subfolders = ['As2', 'As3', 'As4']

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
counts = {}

def show_image_info(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"\nFile: {file_path}")
    print(f"Type: {ext}")
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            print(f"Shape: {data.shape}")
            print(f"Voxel size: {img.header.get_zooms()}")
            plt.imshow(data[:, :, data.shape[2] // 2], cmap='gray')
            plt.title("NIfTI Slice View")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not load NIfTI file: {e}")
    elif ext in image_extensions:
        try:
            img = Image.open(file_path)
            print(f"Size (W x H): {img.size}")
            print(f"Mode: {img.mode}")
            plt.imshow(img)
            plt.title("Image Preview")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not open image file: {e}")

for folder in subfolders:
    folder_path = os.path.join(input_root, folder)
    print(f"\n=== Folder: {folder} ===")
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if file.lower().endswith('.nii.gz') or ext in image_extensions:
                file_path = os.path.join(root, file)
                show_image_info(file_path)
                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break


plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.xlabel('Folder')
plt.ylabel('Number of Images')
plt.title('Image Count per Folder')
plt.tight_layout()
plt.show()

for folder in subfolders:
    folder_path = os.path.join(input_root, folder)
    print(f"\n=== Folder: {folder} ===")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if file.lower().endswith('.nii.gz') or ext in image_extensions:
                file_path = os.path.join(root, file)
                show_image_info(file_path)
                
                
import os
import shutil
import random
from PIL import Image, ImageOps
from collections import defaultdict
import matplotlib.pyplot as plt

input_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS Data\AIIMS Data"
output_root = os.path.join(input_root, "Augmentation")

label_map = {
    'As2': 0,
    'As3': 1,
    'As4': 2
}

reverse_label_map = {v: k for k, v in label_map.items()}
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

data_by_class = defaultdict(list)
final_counts = defaultdict(int)

def is_image(file):
    ext = os.path.splitext(file)[-1].lower()
    return ext in image_extensions

def load_images():
    for folder_name, label in label_map.items():
        folder_path = os.path.join(input_root, folder_name)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if is_image(file):
                    full_path = os.path.join(root, file)
                    data_by_class[label].append(full_path)

def augment_image(image):
    augmentations = []
    augmentations.append(ImageOps.mirror(image))
    augmentations.append(image.rotate(90))
    augmentations.append(image.rotate(180))
    augmentations.append(image.rotate(270))
    return augmentations

def balance_and_save():
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    max_count = max(len(v) for v in data_by_class.values())

    for label, image_paths in data_by_class.items():
        folder_name = reverse_label_map[label]
        output_class_dir = os.path.join(output_root, folder_name)
        os.makedirs(output_class_dir, exist_ok=True)
        for i, img_path in enumerate(image_paths):
            ext = os.path.splitext(img_path)[-1].lower()
            name = f"orig_{i}{ext}"
            shutil.copy(img_path, os.path.join(output_class_dir, name))
            final_counts[label] += 1

        current_count = len(image_paths)
        needed = max_count - current_count

        if needed > 0:
            print(f"Augmenting class '{folder_name}' with {needed} images...")
            i = 0
            while needed > 0:
                img_path = random.choice(image_paths)
                try:
                    image = Image.open(img_path)
                    augs = augment_image(image)
                    for aug in augs:
                        if needed <= 0:
                            break
                        aug_path = os.path.join(output_class_dir, f"aug_{i}.png")
                        aug.save(aug_path)
                        i += 1
                        needed -= 1
                        final_counts[label] += 1
                except Exception as e:
                    print(f"Failed to augment {img_path}: {e}")
            print(f"✓ Augmentation completed for class '{folder_name}'.")


def plot_class_distribution():
    labels = [reverse_label_map[i] for i in sorted(final_counts.keys())]
    values = [final_counts[i] for i in sorted(final_counts.keys())]

    plt.bar(labels, values, color='lightgreen')
    plt.xlabel("Class Folder")
    plt.ylabel("Number of Images After Augmentation")
    plt.title("Balanced Image Count per Class")
    plt.tight_layout()
    plt.show()

load_images()
balance_and_save()
plot_class_distribution()


import os

augmentation_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS Data\AIIMS Data\Augmentation"

label_map = {
    'As2': 0,
    'As3': 1,
    'As4': 2
}

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

dataset = {}

for folder_name, label in label_map.items():
    class_dir = os.path.join(augmentation_root, folder_name)
    for root, _, files in os.walk(class_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in image_extensions:
                full_path = os.path.join(root, file)
                dataset[full_path] = label

print(f"Dataset dictionary created with {len(dataset)} entries.")
print("Sample entries:")
for i, (path, label) in enumerate(dataset.items()):
    if i >= 5:
        break
    print(f"{path} => {label}")
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DEIT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, depth=12, mlp_dim=3072, num_classes=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, embed_dim))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_logits = self.head(x[:, 0])
        dist_logits = self.head_dist(x[:, 1])
        return cls_logits, dist_logits

import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

random.seed(42)
torch.manual_seed(42)

all_samples = list(dataset.items())
random.shuffle(all_samples)
split_idx = int(0.8 * len(all_samples))
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")           
            image = torch.zeros(3, 224, 224)
            label = -1
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_dataset = CustomImageDataset(train_samples, transform=transform)
val_dataset = CustomImageDataset(val_samples, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
import torchvision.models as models
import torch.nn as nn

model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4) 

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4) 
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
num_epochs  = 224
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=running_loss/total, accuracy=100*correct/total)

    train_loss = running_loss / total
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()
