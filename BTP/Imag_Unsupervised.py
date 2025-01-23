import zipfile
import os

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


import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: The file at {image_path} does not exist.")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image at {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    pixels = image.reshape(-1, 3)
    return pixels

def kmeans_clustering(image, k=3):
    pixels = preprocess_image(image)
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
    kmeans.fit(pixels)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape).astype(np.uint8)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f'K-Means Clustering k={k}')
    plt.imshow(segmented_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f'K-Means Scatter Plot k={k}')
    plt.scatter(pixels[:, 0], pixels[:, 1], c=kmeans.labels_, cmap='viridis', s=2)
    plt.xlabel('Red Pixel Value')
    plt.ylabel('Green Pixel Value')
    plt.show()
    
    silhouette = silhouette_score(pixels, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(pixels, kmeans.labels_)
    print(f"K-Means (k={k}) - Silhouette Score: {silhouette}, Davies-Bouldin Index: {davies_bouldin}")
    
    return kmeans

def dbscan_clustering(image, eps=30, min_samples=100):
    pixels = preprocess_image(image)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pixels)
    
    segmented_img = np.zeros_like(image)
    for i in range(len(pixels)):
        label = labels[i]
        if label != -1:
            segmented_img[i // image.shape[1], i % image.shape[1]] = image[i // image.shape[1], i % image.shape[1]]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('DBSCAN Clustering')
    plt.imshow(segmented_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('DBSCAN Scatter Plot')
    plt.scatter(pixels[:, 0], pixels[:, 1], c=labels, cmap='viridis', s=2)
    plt.xlabel('Red Pixel Value')
    plt.ylabel('Green Pixel Value')
    plt.show()
    
    silhouette = silhouette_score(pixels, labels)
    davies_bouldin = davies_bouldin_score(pixels, labels)
    print(f"DBSCAN - Silhouette Score: {silhouette}, Davies-Bouldin Index: {davies_bouldin}")
    
    return labels

def agglomerative_clustering(image, n_clusters=3):
    pixels = preprocess_image(image)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(pixels)
    
    segmented_img = np.zeros_like(image)
    for i in range(len(pixels)):
        label = labels[i]
        segmented_img[i // image.shape[1], i % image.shape[1]] = image[i // image.shape[1], i % image.shape[1]]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f'Agglomerative Clustering k={n_clusters}')
    plt.imshow(segmented_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Agglomerative Scatter Plot k={n_clusters}')
    plt.scatter(pixels[:, 0], pixels[:, 1], c=labels, cmap='viridis', s=2)
    plt.xlabel('Red Pixel Value')
    plt.ylabel('Green Pixel Value')
    plt.show()
    
    silhouette = silhouette_score(pixels, labels)
    davies_bouldin = davies_bouldin_score(pixels, labels)
    print(f"Agglomerative Clustering (k={n_clusters}) - Silhouette Score: {silhouette}, Davies-Bouldin Index: {davies_bouldin}")
    
    return labels

def elbow_method(image_path, max_k=10):
    image = load_image(image_path)
    pixels = preprocess_image(image)
    
    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        kmeans.fit(pixels)
        inertia_values.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), inertia_values, marker='o')
    plt.title('Elbow Method for K-Means')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

def bic_aic_for_kmeans(image, max_k=10):
    pixels = preprocess_image(image)
    bic_values = []
    aic_values = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        kmeans.fit(pixels)
        
        gmm = GaussianMixture(n_components=k)
        gmm.fit(pixels)
        
        bic_values.append(gmm.bic(pixels))
        aic_values.append(gmm.aic(pixels))
    
    plt.plot(range(1, max_k + 1), bic_values, marker='o', label="BIC")
    plt.plot(range(1, max_k + 1), aic_values, marker='x', label="AIC")
    plt.title('BIC/AIC for K-Means')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    best_k_bic = np.argmin(bic_values) + 1
    best_k_aic = np.argmin(aic_values) + 1
    print(f"Optimal k based on BIC: {best_k_bic}, AIC: {best_k_aic}")
    
    return best_k_bic, best_k_aic

def feature_based_clustering(image, k=3):
    pixels = preprocess_image(image)
    
    pca = PCA(n_components=2)
    reduced_pixels = pca.fit_transform(pixels)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reduced_pixels)
    
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape).astype(np.uint8)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f'Feature-based Clustering k={k}')
    plt.imshow(segmented_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Feature-based Scatter Plot k={k}')
    plt.scatter(reduced_pixels[:, 0], reduced_pixels[:, 1], c=kmeans.labels_, cmap='viridis', s=2)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    
    silhouette = silhouette_score(reduced_pixels, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(reduced_pixels, kmeans.labels_)
    print(f"Feature-based Clustering (k={k}) - Silhouette Score: {silhouette}, Davies-Bouldin Index: {davies_bouldin}")
    
    return kmeans

for image_path in image_paths:
    image = load_image(image_path)

    elbow_method(image_path, max_k=10)
    kmeans = kmeans_clustering(image, k=3)
    dbscan_labels = dbscan_clustering(image, eps=30, min_samples=100)
    agglo_labels = agglomerative_clustering(image, n_clusters=3)
    bic_k, aic_k = bic_aic_for_kmeans(image, max_k=10)
    feature_based_clustering(image, k=3)
