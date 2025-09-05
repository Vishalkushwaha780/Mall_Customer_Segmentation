import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('Mall_Customers.csv')

# Plot distributions
def plot_distributions(df):
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    sns.histplot(df['Annual Income (k$)'], kde=True, color='blue', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Annual Income (k$)')
    sns.histplot(df['Age'], kde=True, color='green', ax=axes[0,1])
    axes[0,1].set_title('Distribution of Age')
    sns.histplot(df['Spending Score (1-100)'], kde=True, color='red', ax=axes[1,0])
    axes[1,0].set_title('Distribution of Spending Score')
    df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1,1])
    axes[1,1].set_title('Gender Ratio')
    plt.tight_layout()
    st.pyplot(fig)

# KMeans clustering and plot (2D)
def kmeans_clustering_2d(df, k):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    y_pred = kmeans.fit_predict(X)
    df['Cluster'] = y_pred

    fig, ax = plt.subplots(figsize=(8,6))
    palette = sns.color_palette('bright', k)
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster',
                    data=df, palette=palette, ax=ax)
    ax.set_title(f'Customer Segments with KMeans (k={k})')
    st.pyplot(fig)

    score = silhouette_score(X, y_pred)
    st.write(f'Silhouette Score (2D): {score:.3f}')

# KMeans clustering and plot (3D)
def kmeans_clustering_3d(df, k):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    y_pred = kmeans.fit_predict(X)
    df['Cluster3D'] = y_pred

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    colors = sns.color_palette('bright', k)
    for cluster in range(k):
        cluster_points = df[df['Cluster3D'] == cluster]
        ax.scatter(cluster_points['Annual Income (k$)'],
                   cluster_points['Spending Score (1-100)'],
                   cluster_points['Age'],
                   color=colors[cluster],
                   label=f'Cluster {cluster + 1}',
                   s=50)

    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_zlabel('Age')
    ax.set_title(f'3D Clustering: Income, Spending Score, Age (k={k})')
    ax.view_init(elev=35, azim=185)
    ax.legend()
    st.pyplot(fig)

    score = silhouette_score(X, y_pred)
    st.write(f'Silhouette Score (3D): {score:.3f}')

def main():
    st.title('Mall Customer Segmentation Analysis')
    st.write('This app shows exploratory data analysis and clustering on Mall Customers dataset.')

    df = load_data()
    st.header('Dataset Preview')
    st.dataframe(df.head())

    st.header('Exploratory Data Analysis')
    plot_distributions(df)

    st.header('KMeans Clustering (2D)')
    k = st.slider('Select number of clusters (k)', min_value=2, max_value=10, value=5)
    kmeans_clustering_2d(df, k)

    st.header('KMeans Clustering (3D)')
    kmeans_clustering_3d(df, k)

if __name__ == '__main__':
    main()
