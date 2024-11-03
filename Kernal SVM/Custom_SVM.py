import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import hog
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Initialize variables
X_mitotic = [] 
Y_mitotic = []
X_NON = []
Y_NON = []
saver_dict = {}
saver_mitotic_dict = {}
directory = r"C:\Users\rohan\OneDrive\Desktop\Tester"
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
image_folder = r"C:\Users\rohan\OneDrive\Desktop\Tester"
path = r"C:\Users\rohan\OneDrive\Desktop\Training"
covered_set = set()
svm = SVC(kernel='linear', random_state=42)

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_feature_map(image, bbox):
    x, y, w, h = bbox
    crop_img = image[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (224, 224))
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = preprocess_input(crop_img)
    features = model.predict(crop_img)
    return features

def plot_feature_map(feature_map, bbox_coords):
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < feature_map.shape[-1]:
            ax.imshow(feature_map[0, :, :, i], cmap='viridis')
        ax.axis('off')
    fig.suptitle(f'Feature Maps for Bounding Box {bbox_coords}')
    plt.show()

def annotate_mitotic(saver_mitotic_dict, mitotic_link):
    print("THIS IS THE ANNOTATE FUNCTION in the annotate mitotic ")
    for filename, regions in saver_mitotic_dict.items():
        image_paths = os.path.join(mitotic_link, filename)
        
        if os.path.exists(image_paths):
            try:
                with Image.open(image_paths) as img:
                    draw = ImageDraw.Draw(img)
                    for region in regions:
                        x, y, width, height = region  
                        box = [x, y, x + width, y + height]
                        draw.rectangle(box, outline="green", width=3)
                        
                    img.show()
                    
                    for region in regions:
                        print("file name is ", filename)
                        x, y, width, height = region  
                        box = [x, y, x + width, y + height]
                        draw.rectangle(box, outline="purple", width=3)
                        
                        image = np.array(img)
                        features = extract_feature_map(image, region)
                        
                        image = cv2.imread(image_paths, cv2.IMREAD_GRAYSCALE)
                        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
                        
                        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                        print(fd)
                        print(fd.shape)
                        X_mitotic.append(fd)
                        Y_mitotic.append(1) 
                    
                    save_path = os.path.join(mitotic_link, f"annotated_{filename}")
                    img.save(save_path)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"File {filename} does not exist in the specified folder.")

    return X_mitotic, Y_mitotic

def annotate(saver_dict, path):
    print("THIS IS THE ANNOTATE FUNCTION")
    for filename, regions in saver_dict.items():
        image_path = os.path.join(path, filename)
        
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    draw = ImageDraw.Draw(img)
                    for region in regions:
                        x, y, width, height = region  
                        box = [x, y, x + width, y + height]
                        draw.rectangle(box, outline="purple", width=3)
                        
                    img.show()
                    
                    for region in regions:
                        print("file name is ", filename)
                        x, y, width, height = region  
                        box = [x, y, x + width, y + height]
                        draw.rectangle(box, outline="purple", width=3)
                        
                        image = np.array(img)
                        features = extract_feature_map(image, region)
                        
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
                        
                        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                        print(fd)
                        print(fd.shape)
                        
                        X_NON.append(fd)
                        Y_NON.append(0) 
                    
                    save_path = os.path.join(path, f"annotated_{filename}")
                    img.save(save_path)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"File {filename} does not exist in the specified folder.")
    return X_NON, Y_NON 

def print_mitotic(json_mitotic):
    print("This is the mitotic printer function")
    universal_list  = []
    with open(json_mitotic, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        print(f"File Name: {filename}")
        boundary_box =  []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')
            
            print(f"Bounding Box Coordinates: xmin={xmin}, ymin={ymin}, width={width}, height={height}")
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box =  []
        print("------------------------")
        saver_mitotic_dict[filename] = universal_list
        universal_list = []
        boundary_box = []     
    print(universal_list)
    return saver_mitotic_dict
    
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return fd

def print_filename_bbox(json_file):
    print("This is the printer filename function non-mitotic")
    universal_list  = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    for image_key, image_data in data.items():
        filename = image_data.get('filename', 'Unknown')
        print(f"File Name: {filename}")
        boundary_box =  []
        for region in image_data.get('regions', []):
            shape_attributes = region.get('shape_attributes', {})
            xmin = shape_attributes.get('x', 'N/A')
            ymin = shape_attributes.get('y', 'N/A')
            width = shape_attributes.get('width', 'N/A')
            height = shape_attributes.get('height', 'N/A')
            
            print(f"Bounding Box Coordinates: xmin={xmin}, ymin={ymin}, width={width}, height={height}")
            boundary_box.append(xmin)
            boundary_box.append(ymin)
            boundary_box.append(width)
            boundary_box.append(height)
            universal_list.append(boundary_box)
            boundary_box =  []
        print("------------------------")
        saver_dict[filename] = universal_list
        universal_list = []
        boundary_box = []     
    print(universal_list)
    return saver_dict

# Paths
mitotic_link = r"C:\Users\rohan\OneDrive\Desktop\Train_mitotic"
json_file = 'NonMitotic.json'
json_mitotic = "mitotic.json"

# Process datasets
nums = print_filename_bbox(json_file)
mitotic_output = print_mitotic(json_mitotic)
print("This is the mitotic output file")
print(mitotic_output)
print("This is the Non-mitotic output file")
print(nums)

X_mitotic, Y_mitotic = annotate_mitotic(mitotic_output, mitotic_link)
X_NON, Y_NON = annotate(nums, path)

X = np.array(X_mitotic + X_NON)
y = np.array(Y_mitotic + Y_NON)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
image = Image.open(image_path)
new_image_features = extract_hog_features(image_path)
new_image_features = new_image_features.reshape(1, -1)  
predicted_class = svm_model.predict(new_image_features)

if predicted_class == 1:
    print("The image is predicted as MITOTIC.")
elif predicted_class == 0:
    print("The image is predicted as NON-MITOTIC.")
else:
    print("Unknown prediction.")


import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Mitotic', 'Mitotic'], yticklabels=['Non-Mitotic', 'Mitotic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
