
import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.nii', '.nii.gz'}
input_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS Data"

target_size = (128, 128)

def normalize_image(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return img

def process_image(img):
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_normalized = normalize_image(img_resized)
    return img_normalized

def process_nii_file(filepath):
    img = nib.load(filepath).get_fdata()
    img = np.squeeze(img)
    if img.ndim == 2:
        processed = process_image(img)
    elif img.ndim == 3:
        for i in range(img.shape[2]):
            slice_img = img[:, :, i]
            processed = process_image(slice_img)

def process_image_file(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        processed = process_image(img)

for root, dirs, files in os.walk(input_root):
    rel_path = os.path.relpath(root, input_root)
    for file in tqdm(files, desc=f"Processing {rel_path}"):
        ext = os.path.splitext(file)[-1].lower()
        file_path = os.path.join(root, file)
        if ext in {'.nii', '.nii.gz'}:
            process_nii_file(file_path)
        elif ext in image_extensions:
            process_image_file(file_path)

import os
import matplotlib.pyplot as plt

input_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS Data\AIIMS Data"
subfolders = ['As2', 'As3', 'As4']

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.nii', '.nii.gz'}

counts = {}

for folder in subfolders:
    folder_path = os.path.join(input_root, folder)
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if file.lower().endswith('.nii.gz') or ext in image_extensions:
                count += 1
    counts[folder] = count

plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.xlabel('Folder')
plt.ylabel('Number of Images')
plt.title('Image Count per Folder')
plt.tight_layout()
plt.show()


import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS Data\AIIMS Data"
output_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS_Balanced"
subfolders = ['As2', 'As3', 'As4']
target_size = (128, 128)

def normalize_image(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

def resize_and_normalize(img):
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return normalize_image(img)

def augment_image(img):
    return [
        img,
        np.fliplr(img),
        np.flipud(img),
        np.rot90(img, 1),
        np.rot90(img, 2),
        np.rot90(img, 3)
    ]

if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root)

class_images = {}
image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.nii', '.nii.gz'}

for class_name in subfolders:
    class_path = os.path.join(input_root, class_name)
    images = []
    for root, dirs, files in os.walk(class_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            filename_lower = file.lower()
            if filename_lower.endswith('.nii.gz') or ext in image_exts:
                if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = resize_and_normalize(img)
                        images.append(img)
    class_images[class_name] = images

max_count = max(len(imgs) for imgs in class_images.values())

for class_name, images in class_images.items():
    class_out_dir = os.path.join(output_root, class_name)
    os.makedirs(class_out_dir, exist_ok=True)
    for idx, img in enumerate(images):
        np.save(os.path.join(class_out_dir, f"img_{idx:04d}.npy"), img)
    count = len(images)
    i = 0
    while count < max_count:
        base_img = images[i % len(images)]
        augmented_imgs = augment_image(base_img)
        for aug_img in augmented_imgs:
            if count >= max_count:
                break
            np.save(os.path.join(class_out_dir, f"img_{count:04d}.npy"), aug_img)
            count += 1
        i += 1

balanced_counts = {}
for class_name in subfolders:
    class_out_dir = os.path.join(output_root, class_name)
    balanced_counts[class_name] = len([f for f in os.listdir(class_out_dir) if f.endswith('.npy')])

plt.bar(balanced_counts.keys(), balanced_counts.values(), color='skyblue')
plt.xlabel('Class Folder')
plt.ylabel('Number of Images')
plt.title('Balanced Image Count per Class Folder')
plt.tight_layout()
plt.show()

print(f"Balanced dataset saved at: {output_root}")

import os
import numpy as np
import pickle
from tqdm import tqdm

balanced_root = r"C:\Users\rohan\OneDrive\Desktop\AIIMS_Balanced"
output_dataset_path = r"C:\Users\rohan\OneDrive\Desktop\AIIMS_Balanced\balanced_dataset.pkl"

class_names = sorted([d for d in os.listdir(balanced_root) if os.path.isdir(os.path.join(balanced_root, d))])

all_images = []
all_labels = []

for label, class_name in enumerate(class_names):
    class_folder = os.path.join(balanced_root, class_name)
    npy_files = [f for f in os.listdir(class_folder) if f.endswith('.npy')]
    
    for npy_file in tqdm(npy_files, desc=f"Loading {class_name}"):
        img_path = os.path.join(class_folder, npy_file)
        img = np.load(img_path)
        all_images.append(img)
        all_labels.append(label)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

indices = np.arange(len(all_images))
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

dataset = {
    'images': all_images,
    'labels': all_labels,
    'class_names': class_names
}

with open(output_dataset_path, 'wb') as f:
    pickle.dump(dataset, f)

print(f"Dataset saved at: {output_dataset_path}")
print(f"Samples: {len(all_images)}, Classes: {len(class_names)}")




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = torch.zeros(1, num_patches + 1, embed_dim)
        positions = torch.arange(num_patches + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        self.pos_embedding[:, :, 0::2] = torch.sin(positions * div_term)
        self.pos_embedding[:, :, 1::2] = torch.cos(positions * div_term)

    def forward(self, x):
        return x + self.pos_embedding.to(x.device)

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

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, depth=12, mlp_dim=3072, num_classes=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.classifier(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


class CancerDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = np.stack([img]*3, axis=0).astype(np.float32)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img, label


transform = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_path = r"C:\Users\rohan\OneDrive\Desktop\AIIMS_Balanced\balanced_dataset.pkl"
dataset = CancerDataset(dataset_path, transform=transform)

val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionTransformer(
    img_size=224, patch_size=16, in_channels=3, embed_dim=768,
    num_heads=12, depth=12, mlp_dim=3072, num_classes=len(set(dataset.labels))
).to(device)
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

train_losses = []
val_losses = []
val_accuracies = []

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Val Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
