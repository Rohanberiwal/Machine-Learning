import zipfile
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import cv2

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
import os
import cv2
import matplotlib.pyplot as plt

def get_unlabeled_images(unlabeled_folder):
    unlabeled_image_paths = []

    for file in os.listdir(unlabeled_folder):
        if file.endswith(".png"):
            img_path = os.path.join(unlabeled_folder, file)
            unlabeled_image_paths.append(img_path)

    return unlabeled_image_paths

def get_labeled_images(labeled_folder):
    labeled_image_paths = []

    for file in os.listdir(labeled_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            labeled_image_paths.append(os.path.join(labeled_folder, file))

    return labeled_image_paths

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values in Dataset:")
    print(missing_values[missing_values > 0])
    return missing_values


def show_unlabeled_images(image_paths):
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")
        
        plt.show()

unlabeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\Unlabeled"
labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"

unlabeled_images = get_unlabeled_images(unlabeled_folder_path)
labeled_images = get_labeled_images(labeled_folder_path)

print("Unlabeled Image Paths:")
print(unlabeled_images)

print("\nLabeled Image Paths:")
print(labeled_images)

import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'WHO Grade' not in df.columns:
        raise ValueError("Column 'WHO Grade' not found in CSV file.")
    return df

def plot_scatter(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(df['WHO Grade'])), df['WHO Grade'].astype(str), alpha=0.6, color='blue')
    plt.xlabel("Sample Index")
    plt.ylabel("WHO Grade")
    plt.title("Scatter Plot of WHO Grades")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

import seaborn as sns

def plot_boxplot(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['WHO Grade'])
    plt.title("Boxplot of WHO Grades")
    plt.show()


def plot_bar(df):
    grade_counts = df['WHO Grade'].value_counts()
    plt.figure(figsize=(8, 5))
    grade_counts.plot(kind='bar', color='orange', alpha=0.7)
    plt.xlabel("WHO Grade")
    plt.ylabel("Count")
    plt.title("Distribution of WHO Grades")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()

def detect_outliers(df):
    grades = df['WHO Grade']
    q1 = grades.quantile(0.25)
    q3 = grades.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(grades < lower_bound) | (grades > upper_bound)]
    non_outliers = df[(grades >= lower_bound) & (grades <= upper_bound)]

    return outliers, non_outliers

def check_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"Number of Duplicate Rows: {duplicate_count}")
    return duplicate_count

def plot_outliers(df):
    outliers, non_outliers = detect_outliers(df)

    plt.figure(figsize=(8, 5))
    plt.bar(["Non-Outliers", "Outliers"], [len(non_outliers), len(outliers)], color=['green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Outlier vs Non-Outlier Distribution")
    plt.show()

    print(f"Number of Non-Outliers: {len(non_outliers)}")
    print(f"Number of Outliers: {len(outliers)}")

csv_file_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"
df = load_data(csv_file_path)


#plot_scatter(df)
#plot_bar(df)
#plot_outliers(df)
#check_missing_values(df)
#check_duplicates(df)
#plot_boxplot(df)


import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_case_number(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def check_labeled_images_in_csv(labeled_image_paths, df):
    image_case_numbers = {extract_case_number(path) for path in labeled_image_paths}
    csv_case_numbers = set(df['Case Number'].astype(str))

    matched_cases = image_case_numbers.intersection(csv_case_numbers)
    unmatched_cases = image_case_numbers.difference(csv_case_numbers)

    if unmatched_cases:
        print("Warning: The following labeled images are missing from the CSV:")
        for case in unmatched_cases:
            print(case)
    else:
        print("All labeled images are present in the CSV.")
    
    return matched_cases, unmatched_cases

def plot_pie_chart(matched, unmatched):
    labels = ['Matched', 'Unmatched']
    sizes = [len(matched), len(unmatched)]
    colors = ['#4CAF50', '#FF5733']
    explode = (0.1, 0)
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=140)
    plt.title('Labeled Images: Matched vs Unmatched')
    plt.show()

csv_file_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"
df = pd.read_csv(csv_file_path)

labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"
labeled_images = [os.path.join(labeled_folder_path, file) for file in os.listdir(labeled_folder_path) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

matched, unmatched = check_labeled_images_in_csv(labeled_images, df)
#plot_pie_chart(matched, unmatched)


import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_case_number(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def check_labeled_images_in_csv(labeled_image_paths, df):
    image_case_numbers = {extract_case_number(path) for path in labeled_image_paths}
    csv_case_numbers = set(df['Case Number'].astype(str))

    matched_cases = image_case_numbers.intersection(csv_case_numbers)
    unmatched_cases = image_case_numbers.difference(csv_case_numbers)

    if unmatched_cases:
        print("Warning: The following labeled images are missing from the CSV:")
        for case in unmatched_cases:
            print(case)
    else:
        print("All labeled images are present in the CSV.")
    
    return matched_cases, unmatched_cases

def plot_grade_distribution(df):
    grade_counts = df['WHO Grade'].value_counts()
    
    plt.figure(figsize=(8, 6))
    grade_counts.plot(kind='bar', color=['#4CAF50', '#FF5733', '#FFC107', '#2196F3'])
    plt.xlabel('Grade')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Cases by Grade')
    plt.xticks(rotation=0)
    plt.show()

def plot_images_per_grade(dataset_dir):
    grade_counts = {}
    for grade_folder in os.listdir(dataset_dir):
        grade_path = os.path.join(dataset_dir, grade_folder)
        if os.path.isdir(grade_path):
            grade_counts[grade_folder] = len(os.listdir(grade_path))
    
    plt.figure(figsize=(8, 6))
    plt.bar(grade_counts.keys(), grade_counts.values(), color=['#4CAF50', '#FF5733', '#FFC107', '#2196F3'])
    plt.xlabel('Grade')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Grade')
    plt.xticks(rotation=0)
    plt.show()

def organize_images_by_grade(labeled_image_paths, df, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    for _, row in df.iterrows():
        case_number = str(row['Case Number'])
        grade = f"Grade {row['WHO Grade']}"
        grade_folder = os.path.join(dataset_dir, grade)
        
        if not os.path.exists(grade_folder):
            os.makedirs(grade_folder)
        
        for image_path in labeled_image_paths:
            if extract_case_number(image_path) == case_number:
                new_path = os.path.join(grade_folder, os.path.basename(image_path))
                os.rename(image_path, new_path)


df = pd.read_csv(csv_file_path)
labeled_images = [os.path.join(labeled_folder_path, file) for file in os.listdir(labeled_folder_path) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

matched, unmatched = check_labeled_images_in_csv(labeled_images, df)
#plot_grade_distribution(df)

dataset_seg_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\DatasetSeg"
#organize_images_by_grade(labeled_images, df, dataset_seg_path)
#plot_images_per_grade(dataset_seg_path)

import pandas as pd

def check_grade_counts(df):
    if 'WHO Grade' not in df.columns:
        print("Error: 'WHO Grade' column not found in the DataFrame.")
        return None

    grade_counts = df['WHO Grade'].value_counts().to_dict()
    
    if 1 in grade_counts:
        print(f"Yes, there are {grade_counts[1]} cases of Grade 1.")
    else:
        print("No, there are 0 cases of Grade 1.")

    return grade_counts  

df = pd.read_csv(csv_file_path)
grade_distribution = check_grade_counts(df)

if grade_distribution:
    print("Grade counts:", grade_distribution)

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
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
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

class ImageDataset(Dataset):
    def __init__(self, file_to_grade_dict, transform=None):
        self.file_to_grade_dict = {k: int(v) for k, v in file_to_grade_dict.items() if str(v).isdigit()}
        self.image_paths = list(self.file_to_grade_dict.keys())
        self.labels = list(self.file_to_grade_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img) if self.transform else img
        label = self.labels[idx] - 1 
        return img, label

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def get_file_to_grade_dict(labeled_folder_path, csv_path):
    def normalize_case_name(case_name):
        return case_name.strip().replace('"', '').replace("’", "'").replace("\r", "").replace("\t", "")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    case_to_grade = {}

    for _, row in df.iterrows():
        grade = str(row.get("WHO Grade", "")).strip()
        raw_cases = str(row.get("Case Number", "")).strip().splitlines()
        for case in raw_cases:
            case = normalize_case_name(case)
            if case:
                case_to_grade[case] = grade

    file_to_grade = {}
    for file in os.listdir(labeled_folder_path):
        if file.lower().endswith((".png", ".jpg")):
            full_path = os.path.join(labeled_folder_path, file)
            name_without_ext = os.path.splitext(file)[0]
            matched_grade = case_to_grade.get(name_without_ext, None)
            if matched_grade is None:
                print(f"[!] Grade not found for: {file}")
                matched_grade = "Not Found"
            file_to_grade[full_path] = matched_grade

    return file_to_grade

def show_images_with_grades(file_to_grade_dict):
    for path, grade in file_to_grade_dict.items():
        try:
            image = Image.open(path)
            plt.imshow(image)
            plt.title(f"Grade {grade}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"[!] Could not display {path}: {e}")

labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"
csv_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"

file_to_grade_dict = get_file_to_grade_dict(labeled_folder_path, csv_path)
print(file_to_grade_dict)
grade_count = {}

for grade in file_to_grade_dict.values():
    grade_count[grade] = grade_count.get(grade, 0) + 1

for grade, count in sorted(grade_count.items(), key=lambda x: int(x[0])):
    print(f"Grade {grade}: {count}")
 
import os
import random
import shutil
from collections import defaultdict
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import random
import shutil
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import transforms

labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"
augmented_folder_path = r"C:\Users\rohan\OneDrive\Desktop\aug"
csv_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"

if os.path.exists(augmented_folder_path):
    shutil.rmtree(augmented_folder_path)
os.makedirs(augmented_folder_path)

file_to_grade_dict = get_file_to_grade_dict(labeled_folder_path, csv_path)

grade_to_files = defaultdict(list)
for file, grade in file_to_grade_dict.items():
    grade_to_files[grade].append(file)

target_count = 1000
augmented_file_to_grade_dict = dict(file_to_grade_dict)

for grade, files in grade_to_files.items():
    current_count = len(files)
    needed = target_count - current_count
    if needed <= 0:
        continue

    i = 0
    while i < needed:
        original_path = random.choice(files)
        try:
            image = Image.open(original_path)
            base_name = os.path.basename(original_path)
            name, ext = os.path.splitext(base_name)

            if i % 2 == 0:
                aug_image = ImageOps.mirror(image)
                aug_type = "flip"
            else:
                aug_image = image.rotate(90)
                aug_type = "rot90"

            aug_path = os.path.join(augmented_folder_path, f"{name}_aug{i}_{aug_type}{ext}")
            aug_image.save(aug_path)
            augmented_file_to_grade_dict[aug_path] = grade

            plt.imshow(aug_image)
            plt.title(f"Grade {grade} - Aug Type: {aug_type}")
            plt.axis('off')
            plt.show()

            i += 1

        except Exception as e:
            print(f"Error processing {original_path}: {e}")

grade_count = {}
for grade in augmented_file_to_grade_dict.values():
    grade_count[grade] = grade_count.get(grade, 0) + 1

for grade, count in sorted(grade_count.items(), key=lambda x: int(x[0])):
    print(f"Grade {grade}: {count}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataset = ImageDataset(augmented_file_to_grade_dict, transform=transform)

import matplotlib.pyplot as plt

sorted_grades = sorted(grade_count.items(), key=lambda x: int(x[0]))

grades, counts = zip(*sorted_grades)

plt.bar(grades, counts, color='skyblue')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Distribution of Augmented Images for Each Grade')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def check_key_in_augmented(file_grade_dict, augmented_grad_dict):
    for key in file_grade_dict:
        if key in augmented_grad_dict:
            print("yes")
        else:
            print("no")

check_key_in_augmented(file_to_grade_dict, augmented_file_to_grade_dict)


import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from torchvision import transforms
from transformers import AutoTokenizer

class ImageGradeDataset(Dataset):
    def __init__(self, file_to_grade_dict, processor, transform=None):
        self.file_to_grade_dict = file_to_grade_dict
        self.processor = processor
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.file_to_grade_dict)

    def __getitem__(self, idx):
        img_path, grade = list(self.file_to_grade_dict.items())[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        grade_input = self.tokenizer(grade, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        inputs = self.processor(images=image, return_tensors="pt")
        
        return inputs, grade_input
    
dataset = ImageGradeDataset(augmented_file_to_grade_dict, transform=transform)
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import LayerNorm

def check_dataloader(dataloader, name="Dataloader", num_batches=1):
    print(f"\n--- Checking {name} ---")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")

        if isinstance(batch, (list, tuple)):
            print(f"  Number of items in batch: {len(batch)}")
            for i, item in enumerate(batch):
                if isinstance(item, torch.Tensor):
                    print(f"  Item {i}:")
                    print(f"    Type   : Tensor")
                    print(f"    Shape  : {item.shape}")
                    print(f"    Dtype  : {item.dtype}")
                    print(f"    Numel  : {item.numel()}")
                    if item.dim() != 4:
                        print(f"  Warning: Tensor is not 4D (got {item.dim()}D)")
                else:
                    print(f"  Item {i}: Non-tensor | Type: {type(item)}")

        elif isinstance(batch, dict):
            print(f"  Number of keys in batch: {len(batch)}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  '{key}':")
                    print(f"    Type   : Tensor")
                    print(f"    Shape  : {value.shape}")
                    print(f"    Dtype  : {value.dtype}")
                    print(f"    Numel  : {value.numel()}")
                    if value.dim() == 4:
                        print(f"  '{key}' is a valid 4D tensor")
                    else:
                        print(f"  Warning: '{key}' is not 4D (got {value.dim()}D)")

                else:
                    print(f"  '{key}': Non-tensor | Type: {type(value)}")

            if 'label' in batch:
                print(f"  Labels: {batch['label']}")

        else:
            print(f"  Batch is not list/tuple/dict | Type: {type(batch)}")

        if batch_idx + 1 >= num_batches:
            break

        
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)    

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
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
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

class CrossAttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_classes):
        super().__init__()
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=6
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image_features, text_features):
        B = image_features.shape[0]
        memory = image_features.unsqueeze(0).repeat(B, 1, 1)
        tgt = text_features.unsqueeze(0).repeat(B, 1, 1)
        x = self.transformer(tgt, memory)
        x = self.norm(x)
        return self.classifier(x)

class BLIPModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, depth=12, mlp_dim=3072, num_classes=4):
        super().__init__()
        self.vit = VisionTransformer(img_size, patch_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth, mlp_dim=mlp_dim, num_classes=num_classes)
        self.cross_attention_decoder = CrossAttentionDecoder(embed_dim, num_heads, mlp_dim, num_classes)

    def forward(self, img, text_features):
        image_features = self.vit(img)
        output = self.cross_attention_decoder(image_features, text_features)
        return output

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageDataset(Dataset):
    def __init__(self, file_to_grade_dict, transform=None):
        self.file_to_grade_dict = {k: int(v) for k, v in file_to_grade_dict.items() if str(v).isdigit()}
        self.image_paths = list(self.file_to_grade_dict.keys())
        self.labels = list(self.file_to_grade_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img) if self.transform else img
        label = self.labels[idx] - 1 
        return img, label


dataset = ImageDataset(file_to_grade_dict, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
blip_model = BLIPModel(num_classes=4)


for img, text_features in data_loader:
    img = img.cuda()
    text_features = text_features.cuda() 
    output = blip_model(img, text_features)
    print(output.shape)


dataset = ImageDataset(file_to_grade_dict, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model = BLIPModel(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(blip_model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    blip_model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for img, text_features in data_loader:
        img, text_features = img.to(device), text_features.to(device)

        optimizer.zero_grad()
        output = blip_model(img, text_features)
        
        loss = criterion(output, text_features)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        correct_preds += (predicted == text_features).sum().item()
        total_preds += text_features.size(0)
    
    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_preds / total_preds * 100

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
