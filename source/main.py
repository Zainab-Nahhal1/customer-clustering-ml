import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(path="data/Mall_Customers.csv"):
    return pd.read_csv(path)

def find_optimal_k(X, k_range=range(2, 10)):
    distortions = []
    sil_scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        distortions.append(model.inertia_)
        sil_scores.append(silhouette_score(X, model.labels_))
    return distortions, sil_scores

def plot_elbow(k_range, distortions):
    plt.figure(figsize=(8,4))
    plt.plot(k_range, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def plot_silhouette(k_range, sil_scores):
    plt.figure(figsize=(8,4))
    plt.plot(k_range, sil_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.show()

def cluster_customers(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)
    return clusters, model

def plot_clusters(data, model):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x='Annual_Income_(k$)',
        y='Spending_Score',
        hue='Cluster',
        palette='Set2',
        data=data,
        s=100,
        alpha=0.7
    )
    sns.scatterplot(
        x=model.cluster_centers_[:,0],
        y=model.cluster_centers_[:,1],
        color='red',
        s=300,
        marker='X',
        label='Centroids'
    )
    plt.show()

def main():
    data = load_data()
    X = data[['Annual_Income_(k$)', 'Spending_Score']].values
    distortions, sil_scores = find_optimal_k(X)
    plot_elbow(range(2,10), distortions)
    plot_silhouette(range(2,10), sil_scores)
    best_k = range(2,10)[np.argmax(sil_scores)]
    print(f"✅ Optimal number of clusters (k) = {best_k}")
    data['Cluster'], model = cluster_customers(X, best_k)
    plot_clusters(data, model)

if __name__ == "__main__":
    main()
