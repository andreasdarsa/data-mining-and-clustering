# Data Mining & Clustering Project

**Date:** December 2025  
**Course:** Data Mining  

## 👥 Team Members
* **Georgios Lazaridis**
* **Nicholas Christoforou**
* **Andreas Drasaklis**

---

## 📌 Overview
This project focuses on **unsupervised learning** techniques to clean corrupted datasets, identify outliers, and perform clustering. We implemented a custom **Iterative K-Means** algorithm combined with **Manual Scaling** to accurately detect cluster centers in geometrically complex datasets (e.g., hexagonal formations).

The repository includes scripts for data pre-processing (cleaning) and the main analytical model.

## 🚀 Key Features
* **Defensive Data Cleaning:** Robust handling of corrupted lines, coercion errors, and duplicate removal using Pandas.
* **Manual Scaling:** Custom geometric scaling to normalize $X$ and $Y$ axes for accurate Euclidean distance calculations.
* **Iterative Refinement:** A two-step K-Means approach:
    1.  **Dirty Run:** Initial estimation of clusters.
    2.  **Core Filtering:** Training only on the "core" 30% of points to find refined centroids.
* **Dynamic Outlier Detection:** Using a $3\sigma$ (Standard Deviation) threshold to classify and isolate anomalies.

---

## 📂 Project Structure

```text
├── data/                   # Processed/Cleaned data files
├── raw_data/               # Original corrupted data files
├── plots/                  # Generated visualizations (optional)
├── main.py                 # Main script for Outlier Detection & Clustering
├── cleaner.py              # Script for cleaning corrupted datasets
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
````

## 🛠️ Installation & Setup

It is recommended to use a virtual environment to avoid conflicts.

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/lazoulios/data-mining-and-clustering.git](https://github.com/lazoulios/data-mining-and-clustering.git)
    cd data-mining-and-clustering
    ```

2.  **Create a Virtual Environment (Optional but recommended):**

    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

### 1\. Data Cleaning

To process the raw corrupted files into clean CSVs:

```python
# Run the cleaning function provided in the scripts
clean_dataset('raw_data/data_corrupted.txt', 'data/clean_data.csv')
```

### 2\. Clustering & Outlier Detection

To run the iterative K-Means algorithm:

```bash
python main.py
```

*Note: The script will generate visualization plots showing Normal points, Outliers (Red 'X'), and Refined Centroids (Black Stars).*

## 📈 Results

  * **Hexagonal Clusters:** The method successfully identifies centroids in non-spherical hexagonal layouts.
  * **Outlier Separation:** The $3\sigma$ threshold effectively isolates noise without removing edge-case valid data points.

## 📚 Libraries Used

  * **Pandas:** Data manipulation
  * **NumPy:** Numerical calculations
  * **Matplotlib:** Visualization
  * **Scikit-Learn:** K-Means implementation
