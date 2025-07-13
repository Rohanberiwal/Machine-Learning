import os
import matplotlib.pyplot as plt

plot_output_dir = "/home/madhavimathur/testerViTnew"
os.makedirs(plot_output_dir, exist_ok=True)

input_root = "/home/madhavimathur/AIIMS_datanew"
subfolders = ['AS2', 'AS3', 'AS4']

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

input_root = "/home/madhavimathur/AIIMS_datanew"
output_root = "/home/madhavimathur/AIIMS_balanced"
subfolders = ['AS2', 'AS3', 'AS4']
target_size = (128, 128)
target_count = 30000

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
image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

for class_name in subfolders:
    class_path = os.path.join(input_root, class_name)
    images = []
    for root, _, files in os.walk(class_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in image_exts:
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = resize_and_normalize(img)
                    images.append(img)
    class_images[class_name] = images

for class_name, images in class_images.items():
    class_out_dir = os.path.join(output_root, class_name)
    os.makedirs(class_out_dir, exist_ok=True)
    for idx, img in enumerate(images):
        np.save(os.path.join(class_out_dir, f"img_{idx:05d}.npy"), img)
    count = len(images)
    i = 0
    while count < target_count:
        base_img = images[i % len(images)]
        for aug_img in augment_image(base_img):
            if count >= target_count:
                break
            np.save(os.path.join(class_out_dir, f"img_{count:05d}.npy"), aug_img)
            count += 1
        i += 1

balanced_counts = {}
for class_name in subfolders:
    class_out_dir = os.path.join(output_root, class_name)
    count = len([f for f in os.listdir(class_out_dir) if f.endswith('.npy')])
    balanced_counts[class_name] = count

plt.bar(balanced_counts.keys(), balanced_counts.values(), color='skyblue')
plt.xlabel('Class Folder')
plt.ylabel('Number of Images')
plt.title('Balanced Image Count per Class Folder (Target: 30,000)')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'balanced_image_count.png'))
plt.close()

print(f" Balanced dataset saved at: {output_root}")
print(f"Total Images: {sum(balanced_counts.values())} (Each class: {target_count})")


import os
import numpy as np
import pickle
from tqdm import tqdm

balanced_root = "/home/madhavimathur/AIIMS_balanced"
output_dataset_path = "/home/madhavimathur/AIIMS_balanced/balanced_dataset.pkl"

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


balanced_counts = {c: len(os.listdir(os.path.join(output_root, c))) for c in subfolders}

plt.bar(balanced_counts.keys(), balanced_counts.values(), color='skyblue')
plt.xlabel('Class Folder')
plt.ylabel('Number of Images')
plt.title('Balanced Image Count per Class Folder')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'balanced.png'))
plt.close()

print(f"Balanced dataset saved at: {output_root}")
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
class AIIMSDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.samples = []
        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(class_dir, file), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.load(img_path).astype(np.float32)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).unsqueeze(0)
        return img, label
    
    

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = VisionTransformer(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
    

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms, models

transform = transforms.Compose([transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ranger_adabelief import RangerAdaBelief
import random


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha: {}".format(alpha))
        if not isinstance(k, int) or k < 1:
            raise ValueError("Invalid k: {}".format(k))
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.step_counter = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                state['slow_param'] = p.data.clone()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k != 0:
            return loss

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.optimizer.state[p]
                slow = param_state['slow_param']
                slow += self.alpha * (p.data - slow)
                p.data = slow.clone()
                param_state['slow_param'] = slow

        return loss


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

dataset = datasets.FakeData(transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False, num_classes=10).to(device)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()

criterion = LabelSmoothingCrossEntropy(0.1)
base_optimizer = RangerAdaBelief(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = Lookahead(base_optimizer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
ema_model = models.resnet18(pretrained=False, num_classes=10).to(device)
ema_model.load_state_dict(model.state_dict())
ema_decay = 0.999

def update_ema(model, ema_model, decay):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        if random.random() < 0.5:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(imgs.size()[0]).to(device)
            target_a = labels
            target_b = labels[rand_index]
            imgs = lam * imgs + (1 - lam) * imgs[rand_index]
            outputs = model(imgs)
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(model, ema_model, ema_decay)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, prec, rec, f1, all_labels, all_preds

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, prec, rec, f1, all_labels, all_preds, all_probs

def smooth(values, weight=0.9):
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
    x = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x, smooth(train_losses), label='Train Loss')
    plt.plot(x, smooth(val_losses), label='Val Loss')
    plt.title('Loss'); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(x, smooth(train_accs), label='Train Acc')
    plt.plot(x, smooth(val_accs), label='Val Acc')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(x, smooth(train_f1s), label='Train F1')
    plt.plot(x, smooth(val_f1s), label='Val F1')
    plt.title('F1 Score'); plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png'); plt.close()



def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_pr_curves(y_true, y_probs, num_classes):
    y_true_bin = np.eye(num_classes)[y_true]
    y_probs = np.array(y_probs)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png')
    plt.close()



num_epochs = 50
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc, train_prec, train_rec, train_f1, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_labels, val_preds, val_probs = eval_epoch(ema_model, val_loader, criterion, device)
    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    print(f"Train => Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   => Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
class_names = [str(i) for i in range(model.fc.out_features)]
plot_confusion_matrix(val_labels, val_preds, class_names)
plot_roc_pr_curves(val_labels, val_probs, num_classes=len(class_names))