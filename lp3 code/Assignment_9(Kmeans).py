import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')


# Display the first few rows of the dataset 
print(data.head())

# Check for missing values 
print("Missing values in the dataset:") 
print(data.isnull().sum())

# Select relevant features for clustering (adjust as necessary) 
# Example: using numerical columns only
features = data.select_dtypes(include=[np.number])

# Handle missing values (if any) 
features.fillna(features.mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow method to determine the optimal number of clusters 
inertia = []
K = range(1, 11) 
# Test for 1 to 10 clusters 
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42) 
    kmeans.fit(scaled_features) 
    inertia.append(kmeans.inertia_)

# Plotting the elbow graph 
plt.figure(figsize=(10, 6)) 
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k') 
plt.xlabel('Number of Clusters (k)') 
plt.ylabel('Inertia')
plt.xticks(K) 
plt.grid() 
plt.show()
 
# From the elbow plot, choose the optimal k
optimal_k = 3 # Example, change this based on your elbow plot observation

# Apply K-Means clustering with the optimal number of clusters 
kmeans = KMeans(n_clusters=optimal_k, random_state=42) 
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualizing the clusters (optional, for 2D visualization) 
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=data['Cluster'], cmap='viridis') 
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2') 
plt.colorbar(label='Cluster') 
plt.grid()
plt.show()
