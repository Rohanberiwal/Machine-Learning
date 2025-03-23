import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import zipfile
import os
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Pool, cpu_count


def extract_zip(zip_path, extract_to='/content'):
    if not os.path.exists(zip_path):
        print(f"Error: The file {zip_path} does not exist!")
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("Extracting files...")
            zip_ref.extractall(extract_to)
            print(f"Files successfully extracted to {extract_to}")
    except zipfile.BadZipFile:
        print("Error: The file is not a valid zip file.")

zip_file_path = "/content/Histopathology images.zip"
extract_zip(zip_file_path, extract_to='/content')


def append_images_to_list(folder_path):
    image_list = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    if not os.path.isdir(folder_path):
        print(f"The path '{folder_path}' is not a valid directory.")
        return []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            image_list.append(os.path.join(folder_path, filename))
    return image_list

folder_path  = "/content/Histopathology images"
image_paths  = append_images_to_list(folder_path)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

main_output_folder = "/content/Superpixel_Thing"
os.makedirs(main_output_folder, exist_ok=True)

def superpixel_segmentation(image_path, n_clusters=50):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w, c = image.shape

    X = np.zeros((h * w, 5), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            L, A, B = image[i, j]
            X[i * w + j] = [L, A, B, i / h, j / w]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    segmented_image = labels.reshape(h, w)

    output = np.zeros((h, w, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, (n_clusters, 3))
    for i in range(h):
        for j in range(w):
            output[i, j] = colors[segmented_image[i, j]]

    filename = os.path.basename(image_path)
    image_name, ext = os.path.splitext(filename)
    image_folder = os.path.join(main_output_folder, image_name)
    os.makedirs(image_folder, exist_ok=True)

    save_path = os.path.join(image_folder, f"segmented_{filename}")
    cv2.imwrite(save_path, output)
    plt.figure(figsize=(10, 5))
    plt.imshow(output)
    plt.axis("off")
    plt.title(f"Superpixel Segmentation: {filename}")
    plt.show()

    print(f"Saved in: {image_folder}")

"""
for image_path in image_paths:
    superpixel_segmentation(image_path, n_clusters=100)
"""

def count_superpixel_labels(segmented_folder):
    """Counts the unique labels in each segmented superpixel image."""
    for image_folder in os.listdir(segmented_folder):
        folder_path = os.path.join(segmented_folder, image_folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("segmented_") and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    segmented_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if segmented_image is None:
                        print(f"Error reading {image_path}")
                        continue

                    unique_labels = np.unique(segmented_image)
                    print(f"{filename}: Found {len(unique_labels)} unique labels")
"""
superpixel_folder = "/content/Superpixel_Thing"
count_superpixel_labels(superpixel_folder)
print("end ")
"""

def print_image_sizes(segmented_folder):
    for folder in os.listdir(segmented_folder):
        folder_path = os.path.join(segmented_folder, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("segmented_") and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, filename)
                    img = cv2.imread(image_path)
                    if img is not None:
                        print(f"{filename}: {img.shape}")
                    else:
                        print(f"Error reading {image_path}")
"""
superpixel_folder = "/content/Superpixel_Thing"
print_image_sizes(superpixel_folder)
print("end")
"""
def process_images_parallel(image_folder, n_clusters=50):
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
    with Pool(cpu_count()) as pool:
        results = pool.starmap(superpixel_segmentation, [(img, n_clusters) for img in image_paths])
    print("\n".join([res for res in results if res]))

def count_superpixel_labels(segmented_folder):
    def process_image(image_path):
        segmented_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if segmented_image is None:
            return f"Error reading {image_path}"
        return f"{os.path.basename(image_path)}: {len(np.unique(segmented_image))} unique labels"

    image_paths = [os.path.join(root, file) for root, _, files in os.walk(segmented_folder) for file in files if file.startswith("segmented_") and file.lower().endswith(('.jpg', '.png', '.jpeg'))]
    with Pool(cpu_count()) as pool:
        results = pool.map(process_image, image_paths)
    print("\n".join(results))

def print_image_sizes(segmented_folder):
    def get_size(image_path):
        img = cv2.imread(image_path)
        return f"{os.path.basename(image_path)}: {img.shape}" if img is not None else f"Error reading {image_path}"

    image_paths = [os.path.join(root, file) for root, _, files in os.walk(segmented_folder) for file in files if file.startswith("segmented_") and file.lower().endswith(('.jpg', '.png', '.jpeg'))]
    with Pool(cpu_count()) as pool:
        results = pool.map(get_size, image_paths)
    print("\n".join(results))

image_folder = "/content"
segmented_folder = "/content/Superpixel_Thing"

process_images_parallel(image_folder, n_clusters=100)
count_superpixel_labels(segmented_folder)
print_image_sizes(segmented_folder)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_heads=12, depth=12, mlp_dim=3072):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        x = self.norm(x)
        cls_out = x[:, 0]
        cls_out = F.normalize(cls_out, dim=-1)
        return cls_out

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=256):
        super().__init__()
        self.linear = nn.Linear(input_dim, proj_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.normalize(x, dim=-1)
        return x

class UnsupervisedCLIP(nn.Module):
    def __init__(self, encoder, proj_dim=256):
        super().__init__()
        self.encoder = encoder
        self.proj_head = ProjectionHead(encoder.embed_dim, proj_dim)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.proj_head(feat)
        return proj

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        view1, view2 = self.transform(img)
        return view1, view2

def nt_xent_loss(z, temperature=0.07):
    N = z.shape[0]
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)
    mask = torch.eye(N, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    batch_size = N // 2
    # For the first half, the positive example for sample i is at index i+batch_size.
    logits_first = sim[:batch_size]
    labels_first = torch.arange(batch_size, device=z.device)
    # For the second half, the positive example for sample i is at index i-batch_size.
    logits_second = sim[batch_size:]
    labels_second = torch.arange(batch_size, device=z.device)
    loss_first = F.cross_entropy(logits_first, labels_first)
    loss_second = F.cross_entropy(logits_second, labels_second)
    loss = (loss_first + loss_second) / 2
    return loss

# NT-Xent Loss Function (Fixed for Numerical Stability)
def nt_xent_loss(z, temperature=0.07):
    N = z.shape[0]  # Batch Size * 2
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature  # Similarity matrix
    labels = torch.arange(N, device=z.device)
    loss = F.cross_entropy(sim, labels)
    return loss

# Data Transformations
base_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
transform = TwoCropsTransform(base_transform)


dataset = ImageDataset(image_paths, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VisionTransformer().to(device)
model = UnsupervisedCLIP(encoder, proj_dim=256).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
num_epochs =50
train_losses = []

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for view1, view2 in dataloader:
        view1, view2 = view1.to(device), view2.to(device)
        optimizer.zero_grad()
        proj1, proj2 = model(view1), model(view2)
        features = torch.cat([proj1, proj2], dim=0)
        loss = nt_xent_loss(features, temperature=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Plot Training Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Unsupervised CLIP Contrastive Loss")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Unsupervised CLIP Contrastive Loss")
plt.legend()
plt.grid(True)
plt.show()
