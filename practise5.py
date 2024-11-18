import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Завантаження даних
data = pd.read_csv('Mall_Customers.csv')

# Частина 1: Підготовка даних (EDA)
# Перевірка наявності пропущених значень
print(data.isnull().sum())

# Побудова гістограм розподілу для кожної змінної
data.hist(bins=15, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Розрахунок основних статистичних показників
print(data.describe())

# Стандартизація даних
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Age', 'Annual_Income', 'Spending_Score']])

# Частина 2: Визначення оптимальної кількості кластерів
# Метод ліктя
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.title('Метод ліктя для визначення оптимальної кількості кластерів')
plt.show()

# Розрахунок коефіцієнта силуету
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    silhouette_scores.append(silhouette_score(data_scaled, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Коефіцієнт силуету')
plt.title('Залежність коефіцієнта силуету від кількості кластерів')
plt.show()

# Частина 3: Кластеризація та аналіз результатів
# Виконання кластеризації методом K-means
optimal_k = 5  # Припустимо, що оптимальна кількість кластерів обрана після аналізу
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(data_scaled)
data['Cluster'] = labels

# Візуалізація результатів
plt.figure(figsize=(10, 7))
sns.scatterplot(x=data['Annual_Income'], y=data['Spending_Score'], hue=labels, palette='viridis')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=300, c='red', label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Результати кластеризації K-means')
plt.legend()
plt.show()

# Розрахунок середніх значень показників для кожного кластера
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)

# Частина 4: Додаткові завдання
# Порівняння з іншими методами кластеризації (DBSCAN)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(data_scaled)
data['DBSCAN_Cluster'] = labels_dbscan

plt.figure(figsize=(10, 7))
sns.scatterplot(x=data['Annual_Income'], y=data['Spending_Score'], hue=labels_dbscan, palette='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Результати кластеризації DBSCAN')
plt.show()

# Ієрархічна кластеризація
linkage_matrix = linkage(data_scaled, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix)
plt.title('Дендрограма')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Аналіз стійкості кластерів
silhouette_score_kmeans = silhouette_score(data_scaled, labels)
silhouette_score_dbscan = silhouette_score(data_scaled, labels_dbscan, metric='euclidean')
print('Silhouette Score for K-means:', silhouette_score_kmeans)
print('Silhouette Score for DBSCAN:', silhouette_score_dbscan)
