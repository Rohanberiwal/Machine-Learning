import os
import json
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from torchvision.models import resnet50
from torchvision.ops import roi_pool
import torch.nn.functional as F
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import tensorflow as tf
import os
import json
import os
import json
import numpy as np
from tensorflow.keras.applications import ResNet101 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
standard_dict_mitotic = {}
standard_dict_non_mitotic = {}
import os
from torchvision.ops import roi_pool
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F 
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

floattype = torch.cuda.FloatTensor
import torch

import tensorflow as tf
import numpy as np

def selective_search(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

def get_proposals(image_path):

    image = cv2.imread(image_path)
    proposals = selective_search(image)
    return proposals

def ground_truth_func(json_file):

    with open(json_file, 'r') as f:
        data = json.load(f)
    bounding_boxes = []
    for image_filename, image_data in data.items():
        regions = image_data.get('regions', [])
        for region in regions:
            shape_attributes = region.get('shape_attributes', {})
            x = shape_attributes.get('x', None)
            y = shape_attributes.get('y', None)
            width = shape_attributes.get('width', None)
            height = shape_attributes.get('height', None)
            if x is not None and y is not None and width is not None and height is not None:
                bounding_box = [x, y, x + width, y + height]  
                bounding_boxes.append(bounding_box)
    return bounding_boxes

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.patches as patches
import os
import json
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from torchvision.models import resnet50
from torchvision.ops import roi_pool
import torch.nn.functional as F
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import tensorflow as tf
import os
import json
import os
import json
import numpy as np
from tensorflow.keras.applications import ResNet101 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from torchvision.ops import roi_pool
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import numpy as np
import json
from torchvision.models import resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Function
import pdb
import numpy as np
import torch
import torch.nn.functional as F 
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.patches as patches

root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
mitotic_annotation_file = 'mitotic.json'
non_mitotic_annotation_file = 'NonMitotic.json'
mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch"
non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Non_mitotis_patch"


standard_dict_mitotic = {}
standard_dict_non_mitotic = {}

def print_mitotic(json_mitotic):
    with open(json_mitotic, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            boundary_box.append([xmin, ymin, width, height])  
        standard_dict_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box 

def print_filename_bbox(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        boundary_box = []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')

            boundary_box.append([xmin, ymin, width, height])  # Append the coordinates directly
        standard_dict_non_mitotic[filename.replace('.jpg', '.jpeg')] = boundary_box  # Store boundary boxes directly


def modify_dict_inplace(standard_dict, root):
    keys_to_remove = []
    keys_to_add = []

    for key in standard_dict.keys():
        key_new = key.replace('.jpeg', '.jpg')
        image_key = os.path.join(root, key_new)

        keys_to_remove.append(key)
        keys_to_add.append(image_key)

    for old_key, new_key in zip(keys_to_remove, keys_to_add):
        standard_dict[new_key] = standard_dict.pop(old_key)


def load_annotations_into_dict(annotation_file, root_dir, target_dict):
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        for filename, attributes in data.items():
            img_name = attributes['filename']
            img_path = os.path.join(root_dir, img_name)
            boxes = []
            for region in attributes['regions']:
                shape_attr = region['shape_attributes']
                x = shape_attr['x']
                y = shape_attr['y']
                width = shape_attr['width']
                height = shape_attr['height']
                boxes.append([x, y, width, height])  
            target_dict[img_path] = boxes

    except Exception as e:
        print(f"Error loading annotations from {annotation_file}: {e}")


def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

import cv2
import numpy as np
import tensorflow as tf

def augment_image(image):
    # Convert image to tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Random horizontal and vertical flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.3)  # Increase brightness
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)  # Increase contrast
    
    angle = tf.random.uniform([], minval=-0.3, maxval=0.3)
    image = tf.image.rot90(image, k=tf.cast(angle * 4, tf.int32))
    scale = tf.random.uniform([], minval=0.7, maxval=1.3)
    image = tf.image.resize(image, size=[int(224*scale), int(224*scale)])
    image = tf.image.resize_with_crop_or_pad(image, target_height=224, target_width=224)
    
    return image

def data_generator(file_paths, labels, batch_size=32, augment=False):
    while True:
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            images = []

            for path in batch_paths:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype('float32') / 255.0
                images.append(img)

            images = np.array(images)
            batch_labels = np.array(batch_labels)

            if augment:
                # Apply augmentations to each image
                augmented_images = []
                for image in images:
                    augmented_image = augment_image(image)
                    augmented_images.append(augmented_image)
                
                images = np.array(augmented_images)

            yield images, batch_labels



def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()



def get_output_feature_maps(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    intermediate_model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    output_feature_maps = intermediate_model.predict(image)
    output_feature_maps_np = [np.array(fmap) for fmap in output_feature_maps]
    return output_feature_maps_np

def create_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model



print_mitotic(mitotic_annotation_file)
print_filename_bbox(non_mitotic_annotation_file)
modify_dict_inplace(standard_dict_non_mitotic, root)
modify_dict_inplace(standard_dict_mitotic, root)
load_annotations_into_dict(mitotic_annotation_file, root, standard_dict_mitotic)
load_annotations_into_dict(non_mitotic_annotation_file, root, standard_dict_non_mitotic)

print("Patches saved successfully.")
list_mitotic = get_file_paths(mitotic_save_dir)
list_non_mitotic = get_file_paths(non_mitotic_save_dir)
print("List of mitotic patches:", list_mitotic)
print("List of non-mitotic patches:", list_non_mitotic)
labels_mitotic = [1] * len(list_mitotic)
labels_non_mitotic = [0] * len(list_non_mitotic)
X = np.array(list_mitotic + list_non_mitotic)
y = np.array(labels_mitotic + labels_non_mitotic)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]


split_idx = int(0.7 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


batch_size = 32
train_gen = data_generator(X_train, y_train, batch_size=batch_size, augment=True)
val_gen = data_generator(X_val, y_val, batch_size=batch_size)

history = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=val_gen,
    validation_steps=len(X_val) // batch_size,
    epochs=100
)

plot_training_history(history)
image_path  = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
features = []
featured_map = get_output_feature_maps(model, image_path)

print(featured_map)
    
print("Pipelien ends")


image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
json_file_path = r'C:\Users\rohan\OneDrive\Desktop\Codes\A00.json'
print("CNN is tuned")
print("Starting the code")

bounding_boxes = ground_truth_func(json_file_path)
print("\n********Actual Ground Truth of the Bounding Boxes******")
for idx, box in enumerate(bounding_boxes, start=1):
    print(f'Bounding Box {idx}: {box}')  
print("********Actual Ground Truth of the Bounding Boxes******\n")

print("****** SelectiVE search ********\n")
proposals = get_proposals(image_path) 
print("The proposal regions are:", proposals)
print("Number of Region Proposals:", len(proposals))
print("\n****** SelectiVE search ********\n")

print("****** CNN Layer ********\n")
features = get_output_feature_maps(model, image_path)
print(features)
print("\n****** CNN Layer ********\n")
print("This is the code  for the ROI pooling network")

import torch
import torch.nn as nn

flatten_layer = nn.Flatten()

def convert_and_flatten(feature_maps_list):
    flattened_list = []

    for feature_maps in feature_maps_list:
        if not isinstance(feature_maps, torch.Tensor):
            feature_maps = torch.tensor(feature_maps, dtype=torch.float32)

        if feature_maps.dim() == 4:
            flattened = flatten_layer(feature_maps)
        else:
            flattened = feature_maps.view(feature_maps.size(0), -1)

        batch_size, num_features = flattened.shape
        print(f"Original shape: {flattened.shape}")
        if num_features >= 4:
            new_shape = (batch_size, num_features // 4, 4) if num_features % 4 == 0 else (batch_size, num_features, 1)
            reshaped = flattened.view(new_shape)
        else:
            padding = 4 - num_features % 4
            padded = torch.cat([flattened, torch.zeros(batch_size, padding)], dim=1)
            reshaped = padded.view(batch_size, -1, 4) 
        print(f"Reshaped to: {reshaped.shape}")

        flattened_list.append(reshaped)

    return flattened_list

flattened_list = convert_and_flatten(features)

print(flattened_list)
print("end of the code for the Good")



mitotic_dir = r'C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch'
non_mitotic_dir = r"C:\Users\rohan\OneDrive\Desktop\Non_mitotis_patch"



def folder_iterator(folder_path):
    file_list = []
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                file_list.append((file_name, file_path))
    except Exception as e:
        print(f"An error occurred: {e}")
    for name, path in file_list:
        print(f"File Name: {name}, File Path: {path}")
    
    return file_list

list_mitotic =  folder_iterator(mitotic_dir)
list_non_mitotic = folder_iterator(non_mitotic_dir) 
print(list_mitotic)
print(list_non_mitotic)

def add_labels(image_list, label):
    return [(filename, path, label) for filename, path in image_list]

mitotic_images_labeled = add_labels(list_mitotic ,  1)
non_mitotic_images_labeled = add_labels(list_non_mitotic, 0)

labeled_images = mitotic_images_labeled + non_mitotic_images_labeled
for filename, path, label in labeled_images:
    print(f"Filename: {filename}, Path: {path}, Label: {label}")
    
def create_feature_vector(feature_maps):
    vectors = []
    for fmap in feature_maps[:4]:  
        if len(fmap.shape) > 3:  
            pooled_fmap = np.mean(fmap, axis=(1, 2))  
            vectors.append(pooled_fmap.flatten())
    feature_vector = np.concatenate(vectors)
    return feature_vector

universal_dict = {}

def process_images(labeled_images, model):
    for filename, path, label in labeled_images:
        print(f"Processing: Filename: {filename}, Path: {path}, Label: {label}")
        feature_maps = get_output_feature_maps(model, path)
        feature_vector = create_feature_vector(feature_maps)
        feature_vector_tuple = tuple(feature_vector)
        universal_dict[feature_vector_tuple] = label
        print("Feature Vector:", feature_vector)

process_images(labeled_images , model)
print("Universal Dictionary:")
for feature_vector, label in universal_dict.items():
    print(f"Feature Vector: {feature_vector}, Label: {label}")


def prepare_data_for_svm(universal_dict):
    features = []
    labels = []
    
    for feature_vector, label in universal_dict.items():
        features.append(feature_vector)
        labels.append(label)
    
    return np.array(features), np.array(labels)

X, y = prepare_data_for_svm(universal_dict)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

X_new_scaled = np.array(flattened_list)
predictions = model.predict(X_new_scaled)

print(predictions) 

print("End")










import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def main_regression() :
    # Constants
    mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Mitotis_Patch"
    non_mitotic_save_dir = r"C:\Users\rohan\OneDrive\Desktop\Non_mitotis_patch"
    standard_dict_mitotic = {}
    standard_dict_non_mitotic = {}

    def load_annotations_into_dict(annotation_file, root_dir, target_dict):
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)

            for filename, attributes in data.items():
                img_name = attributes['filename']
                img_path = os.path.join(root_dir, img_name)
                boxes = []
                for region in attributes['regions']:
                    shape_attr = region['shape_attributes']
                    x = shape_attr['x']
                    y = shape_attr['y']
                    width = shape_attr['width']
                    height = shape_attr['height']
                    boxes.append([x, y, width, height])
                target_dict[img_path] = boxes

        except Exception as e:
            print(f"Error loading annotations from {annotation_file}: {e}")

    # Load annotations into dictionaries
    root = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
    mitotic_annotation_file = 'mitotic.json'
    non_mitotic_annotation_file = 'NonMitotic.json'

    load_annotations_into_dict(mitotic_annotation_file ,root, standard_dict_mitotic)
    load_annotations_into_dict(non_mitotic_annotation_file  , root, standard_dict_non_mitotic)

    # Convert keys in dictionaries to match image file extensions
    def modify_dict_inplace(standard_dict, root):
        keys_to_remove = []
        keys_to_add = []

        for key in standard_dict.keys():
            key_new = key.replace('.jpeg', '.jpg')
            image_key = os.path.join(root, key_new)

            keys_to_remove.append(key)
            keys_to_add.append(image_key)

        for old_key, new_key in zip(keys_to_remove, keys_to_add):
            standard_dict[new_key] = standard_dict.pop(old_key)

    modify_dict_inplace(standard_dict_non_mitotic, root)
    modify_dict_inplace(standard_dict_mitotic, root)
    print(standard_dict_mitotic)
    print("\n")
    print(standard_dict_non_mitotic)


    # Load file paths
    def get_file_paths(directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths

    list_mitotic = get_file_paths(mitotic_save_dir)
    list_non_mitotic = get_file_paths(non_mitotic_save_dir)
    print("List of mitotic patches:", list_mitotic)
    print("List of non-mitotic patches:", list_non_mitotic)

    # Split data into train and validation sets
    train_ratio = 0.7
    validation_ratio = 0.3

    num_mitotic = len(list_mitotic)
    num_non_mitotic = len(list_non_mitotic)

    num_mitotic_train = int(train_ratio * num_mitotic)
    num_mitotic_val = num_mitotic - num_mitotic_train

    num_non_mitotic_train = int(train_ratio * num_non_mitotic)
    num_non_mitotic_val = num_non_mitotic - num_non_mitotic_train

    mitotic_train_paths = list_mitotic[:num_mitotic_train]
    mitotic_val_paths = list_mitotic[num_mitotic_train:]

    non_mitotic_train_paths = list_non_mitotic[:num_non_mitotic_train]
    non_mitotic_val_paths = list_non_mitotic[num_non_mitotic_train:]

    print("Number of mitotic patches (train):", num_mitotic_train)
    print("Number of mitotic patches (val):", num_mitotic_val)
    print("Number of non-mitotic patches (train):", num_non_mitotic_train)
    print("Number of non-mitotic patches (val):", num_non_mitotic_val)

    # Helper function to load image data and labels
    def load_image_and_label(image_path, label, target_size=(224, 224)):
        try:
            image = load_img(image_path, target_size=target_size)
            image_array = img_to_array(image)
            image_array = image_array / 255.0
            return image_array, label

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None

    # Prepare training data for classification
    train_data = []
    train_labels = []

    for image_path in mitotic_train_paths:
        image_array, label = load_image_and_label(image_path, 1)
        if image_array is not None:
            train_data.append(image_array)
            train_labels.append(label)

    for image_path in non_mitotic_train_paths:
        image_array, label = load_image_and_label(image_path, 0)
        if image_array is not None:
            train_data.append(image_array)
            train_labels.append(label)

    # Prepare validation data for classification
    val_data = []
    val_labels = []

    for image_path in mitotic_val_paths:
        image_array, label = load_image_and_label(image_path, 1)
        if image_array is not None:
            val_data.append(image_array)
            val_labels.append(label)

    for image_path in non_mitotic_val_paths:
        image_array, label = load_image_and_label(image_path, 0)
        if image_array is not None:
            val_data.append(image_array)
            val_labels.append(label)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)

    print("Training data shape:", train_data.shape)
    print("Training labels shape:", train_labels.shape)
    print("Validation data shape:", val_data.shape)
    print("Validation labels shape:", val_labels.shape)

    # Build a simple CNN model for classification
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the classification model
    history = model.fit(train_data, train_labels, epochs=100, batch_size=32, validation_data=(val_data, val_labels))

    # Evaluate the classification model
    loss, accuracy = model.evaluate(val_data, val_labels)
    print(f"Validation loss (classification): {loss:.4f}")
    print(f"Validation accuracy (classification): {accuracy:.4f}")

    # Plot classification model training history
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy (Classification)')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss (Classification)')

    plt.tight_layout()
    plt.show()

    print("This is the train phase")
    train_bb_data = []
    train_bb_targets = []

    # Process mitotic images
    for image_path, boxes in standard_dict_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                # Normalize coordinates relative to image size
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                train_bb_data.append(image_array)
                train_bb_targets.append(target)

    # Process non-mitotic images
    for image_path, boxes in standard_dict_non_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                train_bb_data.append(image_array)
                train_bb_targets.append(target)


    print("This is hte validation Phase")
    val_bb_data = []
    val_bb_targets = []

    # Process mitotic images
    for image_path, boxes in standard_dict_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                # Normalize coordinates relative to image size
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                val_bb_data.append(image_array)
                val_bb_targets.append(target)

    # Process non-mitotic images
    for image_path, boxes in standard_dict_non_mitotic.items():
        image_array, _ = load_image_and_label(image_path, None)  # Load image without label
        if image_array is not None:
            for box in boxes:
                xmin, ymin, width, height = box
                # Normalize coordinates relative to image size
                target = [xmin / image_array.shape[1], ymin / image_array.shape[0], width / image_array.shape[1], height / image_array.shape[0]]
                val_bb_data.append(image_array)
                val_bb_targets.append(target)

    # Convert to numpy arrays
    train_bb_data = np.array(train_bb_data)
    train_bb_targets = np.array(train_bb_targets)
    val_bb_data = np.array(val_bb_data)
    val_bb_targets = np.array(val_bb_targets)

    print("Bounding Box Regression - Training data shape:", train_bb_data.shape)
    print("Bounding Box Regression - Training targets shape:", train_bb_targets.shape)
    print("Bounding Box Regression - Validation data shape:", val_bb_data.shape)
    print("Bounding Box Regression - Validation targets shape:", val_bb_targets.shape)

    # Define and compile the bounding box regression model
    bb_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='linear')  # Output 4 for (xmin, ymin, width, height)
    ])

    bb_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the BB regx model
    bb_history = bb_model.fit(train_bb_data, train_bb_targets, epochs= 50 , validation_data=(val_bb_data, val_bb_targets))

    # Evaluate the BB regx model
    bb_loss, bb_mae = bb_model.evaluate(val_bb_data, val_bb_targets)
    print(f"BB regx model evaluation - Loss: {bb_loss}, MAE: {bb_mae}")

    # Plot BB regx model training history
    plt.plot(bb_history.history['loss'], label='Train Loss')
    plt.plot(bb_history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Bounding Box Regression - Training and Validation Loss')

    plt.tight_layout()
    plt.show()

    # Fit BB regressor model
    bb_history = bb_model.fit(train_bb_data, train_bb_targets, epochs=50, validation_data=(val_bb_data, val_bb_targets))

    # Evaluate BB regressor model on training data
    train_loss, train_mae = bb_model.evaluate(train_bb_data, train_bb_targets)
    print(f"BB regressor model - Training MAE: {train_mae}")

    # Evaluate BB regressor model on validation data
    val_loss, val_mae = bb_model.evaluate(val_bb_data, val_bb_targets)
    print(f"BB regressor model - Validation MAE: {val_mae}")

    # Plot BB model training history
    plt.plot(bb_history.history['mae'], label='Train MAE')
    plt.plot(bb_history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('BB Regressor Model Training MAE')
    plt.show()

#main_regression()
#print("This  is where the code starts")
