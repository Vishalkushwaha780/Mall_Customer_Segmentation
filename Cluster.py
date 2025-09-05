import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    return pd.read_csv('Mall_Customers.csv')

@st.cache_data
def perform_kmeans(data, clusters, features):
    kmeans = KMeans(n_clusters=clusters, init='k-means++')
    kmeans.fit(data[features])
    data['Cluster'] = kmeans.labels_
    return data

def plot_clusters(data, features):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features[0], y=features[1], hue='Cluster', palette='viridis', data=data)
    plt.title('Customer Segments')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    st.pyplot(plt)

def main():
    st.title('Mall Customer Segmentation')
    st.write('Segment mall customers using KMeans clustering.')

    df = load_data()
    st.dataframe(df.head(20))

    features = st.multiselect('Select features for clustering',
                              ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
                              default=['Annual Income (k$)', 'Spending Score (1-100)'])
    clusters = st.slider('Number of clusters (K)', 2, 10, 5)

    if st.button('Segment Customers'):
        clustered_data = perform_kmeans(df.copy(), clusters, features)
        st.write('Clustered Data Sample:')
        st.dataframe(clustered_data.head())
        plot_clusters(clustered_data, features)

if __name__ == '__main__':
    main()
