# Mall Customer Segmentation Using Clustering Techniques

## Project Overview
This project performs customer segmentation analysis on a Mall Customers dataset using machine learning clustering algorithms. The primary goal is to identify distinct customer groups based on their annual income, spending score, and age, to help develop targeted marketing strategies that improve sales and customer engagement.

## Dataset
The dataset contains information about 200 mall customers with the following features:
- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income in thousand dollars
- **Spending Score (1-100)**: Score assigned by the mall based on customer spending behavior

## Technologies Used
- Python 3
- [Streamlit](https://streamlit.io/) for building the interactive web application
- Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Scikit-learn for clustering algorithms and evaluation metrics (KMeans, Silhouette Score)

## Features
- Exploratory Data Analysis (EDA) with distribution plots of key variables
- Interactive KMeans clustering:
  - Select the number of clusters via a slider
  - Visualize clusters in 2D (Annual Income vs Spending Score)
  - Visualize clusters in 3D (Annual Income, Spending Score, Age)
  - Display Silhouette Scores for clustering quality evaluation

## How to Run
1. Clone this repository:
2. Install the required packages:
3. Run the Streamlit app:
4. 
4. The app will open in your browser, where you can explore the data and interact with clustering options.

---

## Screenshots

![EDA Plots](eda_screenshot.png)  
*Exploratory Data Analysis showing distributions*

![2D Clustering](2d_cluster_screenshot.png)  
*2D KMeans Clustering visualization*

![3D Clustering](3d_cluster_screenshot.png)  
*3D KMeans Clustering visualization*

---

## Future Work
- Integrate DBSCAN clustering option with parameter tuning
- Add customer segmentation insights explanations per cluster
- Add download option for clustered dataset
- Deploy Streamlit app on cloud platforms for easy access

---

## Author
- Vishal Kushwaha  
[GitHub Profile](https://github.com/Vishalkushwaha780)

---

## License
This project is licensed under the MIT License.

---
