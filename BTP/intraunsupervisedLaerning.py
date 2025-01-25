import zipfile
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import measure
import matplotlib.pyplot as plt

def extract_and_list_files(zip_file_path, extract_to_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)

    extracted_files = []
    for root, dirs, files in os.walk(extract_to_folder):
        for file in files:
            extracted_files.append(os.path.join(root, file))

    return extracted_files

def extract_feature_map(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    feature_map = h
    return feature_map

def apply_kmeans(image, n_clusters=3):
    feature_map = extract_feature_map(image)
    pixels = feature_map.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    segmented_image = kmeans.labels_.reshape(image.shape[:2])
    return segmented_image, kmeans.labels_

def apply_canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    return edges

def apply_watershed(image, feature_map):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature_map_8bit = cv2.convertScaleAbs(feature_map)
    _, thresh = cv2.threshold(feature_map_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image, markers


def count_components(image):
    labeled_image, num_labels = measure.label(image, connectivity=2, return_num=True)
    return num_labels, labeled_image

def process_images(zip_file_path, extract_to_folder, output_folder):
    image_paths = extract_and_list_files(zip_file_path, extract_to_folder)
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        if image_path.endswith(".jpeg") or image_path.endswith(".png"):
            print(f"Processing image: {os.path.basename(image_path)}")
            image = cv2.imread(image_path)

            kmeans_segmented, _ = apply_kmeans(image, n_clusters=3)
            kmeans_components, _ = count_components(kmeans_segmented)
            print(f"  K-Means detected components: {kmeans_components}")

            canny_edges = apply_canny(image)
            canny_components, _ = count_components(canny_edges)
            print(f"  Canny detected edge components: {canny_components}")

            watershed_result, watershed_markers = apply_watershed(image.copy(), kmeans_segmented)
            watershed_components, _ = count_components(watershed_markers)
            print(f"  Watershed detected regions: {watershed_components}")

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(kmeans_segmented, cmap='tab20b')
            plt.title("K-Means Segmentation")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(canny_edges, cmap='gray')
            plt.title("Canny Edge Detection")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(watershed_result, cv2.COLOR_BGR2RGB))
            plt.title("Watershed Segmentation")
            plt.axis('off')
            plt.show()

            output_path_kmeans = os.path.join(output_folder, f"kmeans_{os.path.basename(image_path)}")
            output_path_canny = os.path.join(output_folder, f"canny_{os.path.basename(image_path)}")
            output_path_watershed = os.path.join(output_folder, f"watershed_{os.path.basename(image_path)}")

            cv2.imwrite(output_path_kmeans, kmeans_segmented)
            cv2.imwrite(output_path_canny, canny_edges)
            cv2.imwrite(output_path_watershed, watershed_result)

zip_file_path = '/content/Histopathology images.zip'
extract_to_folder = '/content/Histopathology_images_extracted/'
output_folder = '/content/output_images/'
process_images(zip_file_path, extract_to_folder, output_folder)
