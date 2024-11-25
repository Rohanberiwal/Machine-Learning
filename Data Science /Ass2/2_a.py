import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd

data = fetch_kddcup99(as_frame=True)
X = data.data
y = data.target

X = X.apply(pd.to_numeric, errors='ignore')

numeric_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(exclude=['number']).columns

if len(numeric_cols) > 0:
    numeric_transformer = SimpleImputer(strategy='mean')
    X[numeric_cols] = numeric_transformer.fit_transform(X[numeric_cols])

label_encoders = {}
if len(categorical_cols) > 0:
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def compute_kmeans_loss_with_centroids(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    closest_centroids = np.argmin(distances, axis=1)
    inertia = np.sum(np.min(distances, axis=1))
    return inertia

def random_projection_experiment(X, n_components=20, k=15, n_repeats=5):
    results = []

    for i in range(n_repeats):
        transformer = GaussianRandomProjection(n_components=n_components, random_state=42)
        X_transformed = transformer.fit_transform(X)

        kmeans_original = KMeans(n_clusters=k, random_state=42)
        kmeans_original.fit(X)
        centroids_original = kmeans_original.cluster_centers_

        kmeans_transformed = KMeans(n_clusters=k, random_state=42)
        kmeans_transformed.fit(X_transformed)
        centroids_transformed = kmeans_transformed.cluster_centers_

        loss_original_with_B = compute_kmeans_loss_with_centroids(X, centroids_original)
        loss_transformed_with_A = compute_kmeans_loss_with_centroids(X_transformed, centroids_transformed)

        results.append([loss_original_with_B, loss_transformed_with_A])

    return np.array(results)

results = random_projection_experiment(X_scaled, n_components=20, k=15, n_repeats=5)

df = pd.DataFrame(results, columns=['Loss on  original data' , 'Loss on  transformed data',])
print(df)

df.plot(kind='bar', figsize=(10, 6))
plt.title("K-means Loss on Original vs Transformed Data (Inertia) ")
plt.ylabel("K-means Loss (Inertia)")
plt.xticks(rotation=0)
plt.show()
