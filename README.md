# Mall Customers Clustering Project

## Description
This project performs customer segmentation on the Mall Customers dataset using K-Means Clustering. The goal is to group customers based on annual income and spending score. This allows businesses to better understand customer behavior and tailor marketing strategies.

## Features
- Determine the optimal number of clusters using Elbow Method and Silhouette Score.
- Visualize clusters and centroids.
- Clear and modular Python code for easy adaptation.

## How to Run
1. Install Python 3.x and required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Place the dataset `Mall_Customers.csv` in the `data/` folder.
3. Run the main script:
   ```bash
   python source/main.py
   ```
4. The script will display cluster visualizations and print the optimal number of clusters.

## Project Structure
- `source/` : Contains the main Python code (`main.py`).
- `data/` : Contains the original dataset.
- `tests/` : Contains unit tests for the project.
- `sample/` : Contains a small sample dataset for testing.
- `.gitignore` : Specifies files to ignore in Git.
- `Makefile` : Automates common tasks like running the script or tests.
